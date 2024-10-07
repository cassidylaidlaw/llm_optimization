from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torchtyping import TensorType
from trlx.data.method_configs import register_method
from trlx.models.modeling_ppo import PPOConfig
from trlx.utils.modeling import flatten_dict


@dataclass
@register_method
class PPOWithRefPolicyConfig(PPOConfig):
    def loss(
        self,
        logprobs: TensorType["batch_size", "response_size"],
        values: TensorType["batch_size", "response_size"],
        old_logprobs: TensorType["batch_size", "response_size"],
        ref_logprobs: TensorType["batch_size", "response_size"],
        old_values: TensorType["batch_size", "response_size"],
        advantages: TensorType["batch_size", "response_size"],
        returns: TensorType["batch_size", "response_size"],
        mask: TensorType["batch_size", "response_size"],
        kl_weight: Optional[float] = None,
        sqrt_chi2_weight: Optional[float] = None,
    ):
        loss, stats = super().loss(
            logprobs,
            values,
            old_logprobs,
            old_values,
            advantages,
            returns,
            mask,
        )

        log_ratio = ((logprobs - ref_logprobs) * mask).sum(1)

        log_ratio_clip = 10

        ratio_m_1 = torch.where(
            log_ratio >= log_ratio_clip,
            np.exp(log_ratio_clip) * (1 + log_ratio - log_ratio_clip) - 1,
            torch.special.expm1(log_ratio.clamp(max=log_ratio_clip)),
        )
        inv_ratio_m_1 = torch.where(
            log_ratio <= -log_ratio_clip,
            np.exp(log_ratio_clip) * (1 - log_ratio - log_ratio_clip) - 1,
            torch.special.expm1(-log_ratio.clamp(min=-log_ratio_clip)),
        )

        kl = (log_ratio + inv_ratio_m_1).mean()
        chi2 = (ratio_m_1 + inv_ratio_m_1).mean()

        stats["regularization"] = {
            "kl": kl.item(),
            "chi2": chi2.item(),
            "ratio_clipfrac": (log_ratio > log_ratio_clip).float().mean().item(),
            "inv_ratio_clipfrac": (log_ratio < -log_ratio_clip).float().mean().item(),
        }

        if kl_weight is not None:
            loss = loss + kl_weight * kl
        if sqrt_chi2_weight is not None:
            loss = loss + sqrt_chi2_weight * torch.sqrt(chi2.clip(min=1e-4))

        return loss, flatten_dict(stats)
