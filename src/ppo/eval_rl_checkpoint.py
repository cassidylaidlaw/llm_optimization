# Taken and modified from Open-Assistant's model/model_training/trainer_rl.py

import json
import os
from argparse import Namespace
from typing import List

import tqdm
import transformers
import trlx
from datasets import Dataset
from model_training.custom_datasets.formatting import format_pairs
from model_training.utils.utils import init_rng
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trlx.data.configs import TRLConfig

from src.data_utils.oa_custom_datasets.get_dataset_patch import get_dataset
from src.ppo.trainer_rl import argument_parsing
from src.reward_modeling.scoring.score import score_answers


def main():
    training_conf = argument_parsing(
        checkpoint="",
        batch_size=8,
        gold_model="allenai/llama-3-tulu-2-8b-uf-mean-rm",
        output="",
    )
    rank_config = Namespace(**training_conf.rank_config)
    sft_config = Namespace(**training_conf.sft_config)
    gold_config = Namespace(**training_conf.gold_config)
    training_conf.num_eval_prompts = training_conf.eval_size

    init_rng(training_conf)

    eos_token = transformers.AutoTokenizer.from_pretrained(
        sft_config.model_name, cache_dir=sft_config.cache_dir
    ).eos_token

    # Load pretrained SFT model

    # override model_name to be the same as sft_model
    trlx_config = TRLConfig.load_yaml("configs/ppo_config.yaml")
    trlx_config.sft_config = sft_config

    train, eval_dict = get_dataset(training_conf, mode="rl")

    # take the dataset as the eval prompt generation dataset
    eval = eval_dict[next(iter(eval_dict))]

    # trlx requires training data to be a list of prompts
    # first element of each sample is the context and the prompt
    prompts, eval_prompts = tuple(
        map(
            lambda x: [
                "".join(format_pairs(x[i], eos_token, add_initial_reply_token=True))
                for i in range(len(x))
            ],
            (train, eval),
        )
    )

    trlx_config.tokenizer.tokenizer_path = sft_config.model_name
    trlx_config.model.model_path = sft_config.model_name

    trainer = trlx.trainer.accelerate_ppo_trainer.AcceleratePPOTrainer(
        config=trlx_config
    )
    trainer.accelerator._optimizers = []
    trainer.accelerator._schedulers = []
    if training_conf.checkpoint:
        print(f"Loading checkpoint from {training_conf.checkpoint}...")
        trainer.load(training_conf.checkpoint)

    trainer.model.eval()

    batch_size = training_conf.batch_size
    responses: List[str] = []
    answers_ids = []
    for i in tqdm.trange(0, len(eval_prompts), batch_size, desc="Generating responses"):
        responses_ids = trainer.generate(
            **trainer.tokenizer(
                eval_prompts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
            ),
            do_sample=True,
        )
        for response_ids in responses_ids:
            response_ids = response_ids.tolist()
            answers_ids.append([response_ids])
            response_ids = [
                token_id
                for token_id in response_ids
                if token_id != trainer.tokenizer.pad_token_id
            ]
            response = trainer.tokenizer.decode(response_ids)
            responses.append(response)

    instructions = []
    answers = []
    for prompt, response in zip(eval_prompts, responses):
        response = response[len(prompt) :].replace("<|endoftext|>", "")
        prompt = prompt[len("<|prompter|>") : -len("<|endoftext|><|assistant|>")]
        instructions.append(prompt)
        answers.append([response])

    dataset = Dataset.from_dict({
        "instruction": instructions,
        "answers": answers,
        "answers_ids": answers_ids,
    })

    print("Scoring the generated responses with the proxy...")
    dataset = Dataset.from_list(
        score_answers(
            model_name=rank_config.model_names[0],
            dataset=dataset,
            scores_type="proxy_scores",
            sort=False,
            split_size=batch_size,
            is_alpacafarm_rm=False,
        )
    )

    print(f"Loading gold reward model from {training_conf.gold_model}...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_conf.gold_model,
    ).cuda()
    reward_model.eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(training_conf.gold_model)

    gold_scores = []
    for data in tqdm.tqdm(dataset, desc="Gold scoring"):
        (answer,) = data["answers"]
        example = f"""<|user|>
        {data['instruction']}
        <|assistant|>"
        {answer}"""
        inputs = reward_tokenizer(
            example, return_tensors="pt", padding=True, truncation=True
        )
        reward = reward_model(
            **{key: value.cuda() for key, value in inputs.items()}
        ).logits.item()
        gold_scores.append([reward])

    dataset = dataset.add_column("gold_scores", gold_scores)

    results_fname = training_conf.output
    if not results_fname:
        results_fname = os.path.join(training_conf.checkpoint, "gold_eval_with_response_ids.json")

    print(f"Saving results to {results_fname}...")
    with open(results_fname, "w") as results_file:
        json.dump(dataset.to_list(), results_file, indent=2)


if __name__ == "__main__":
    main()
