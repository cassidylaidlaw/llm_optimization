import json
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtyping import TensorType
from trlx.pipeline import BaseRolloutStore


@dataclass
class PPOWithRefPolicyRLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over the response tokens generated
                    by the policy network (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens.
    :type logprobs: torch.Tensor

    :param ref_logprobs: The log probabilities over the response tokens generated
                    by the reference policy network.
    :type ref_logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    logprobs: TensorType["response_size"]
    ref_logprobs: TensorType["response_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPOWithRefPolicyRLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param ref_logprobs: A batch of log probabilities from reference policy
    :type ref_logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size"]
    ref_logprobs: TensorType["batch_size", "response_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]


def ppo_with_ref_policy_collate_fn(
    padding_side: str, pad_token_id: int, elems: Iterable[PPOWithRefPolicyRLElement]
):
    if padding_side == "left":
        # Left padding of already left-padded queries
        query_tensors = pad_sequence(
            [elem.query_tensor.flip(0) for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        ).flip(1)
    elif padding_side == "right":
        query_tensors = pad_sequence(
            [elem.query_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        )
    else:
        raise ValueError(f"Invalid padding side: {padding_side}")

    return PPOWithRefPolicyRLBatch(
        query_tensors,
        # Right pad the rest, to have a single horizontal query/response split
        pad_sequence(
            [elem.response_tensor for elem in elems],
            padding_value=pad_token_id,
            batch_first=True,
        ),
        pad_sequence(
            [elem.logprobs for elem in elems],
            padding_value=0.0,
            batch_first=True,
        ),
        pad_sequence(
            [elem.ref_logprobs for elem in elems], padding_value=0.0, batch_first=True
        ),
        pad_sequence(
            [elem.values for elem in elems], padding_value=0.0, batch_first=True
        ),
        pad_sequence(
            [elem.rewards for elem in elems],
            padding_value=0.0,
            batch_first=True,
        ),
    )


class PPOWithRefPolicyRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[PPOWithRefPolicyRLElement] = [None]

    def push(self, exps: Iterable[PPOWithRefPolicyRLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str, only_text=True):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        def filter_text(d, only_text):
            if only_text:
                keys = list(d.keys())
                for key in keys:
                    if key != "query_tensor" and key != "response_tensor":
                        d.pop(key)
            return d

        data = [filter_text(exp_to_dict(exp), only_text) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPOWithRefPolicyRLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=shuffle,
            collate_fn=partial(
                ppo_with_ref_policy_collate_fn, self.padding_side, self.pad_token_id
            ),
        )
