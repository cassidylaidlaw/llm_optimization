# Preventing RLHF reward hacking with $\chi^2$ regularization

This repository contains the code for the LLM RLHF experiments in the paper [Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking](https://arxiv.org/abs/2403.03185). Our main codebase for running the rest of the experiments is located at https://github.com/cassidylaidlaw/orpo.

This codebase is based off of the code for [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743) by Thomas Coste et al., located at https://github.com/tlc4418/llm_optimization.

## Installation
```
git clone https://github.com/cassidylaidlaw/llm_optimization.git
cd llm_optimization
pip install -e .
```

## Running experiments

### Training a reward model as the proxy

To reproduce our results on RLHF, the first step is to train a reward model based on Coste et al.'s code:

    accelerate launch --config_file configs/accelerate_config.yaml src/sft/trainer_sft.py --configs defaults rm-pythia-44m

This reward model is used as the proxy reward for RL training.

### RLHF

To fine-tune LLMs with PPO, run the following command once the reward model is trained:

    accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf pythia_rlhf_individual --use_sqrt_chi2=True --regularize_in_loss=True --coeff=0.0008 --rng_seed=0

To train with KL divergence, change `--use_sqrt_chi2=True` to `--use_sqrt_chi2=False`. To train without regularization, set `--coeff=0`.

### Evaluating trained models

To evaluate the proxy and true rewards of trained LLMs, run the following command:

    python -m src.ppo.eval_rl_checkpoint --checkpoint runs/ppo_individual/models_rm-pythia-44m_seed0/sqrt_chi2_in_loss-0.0008/seed_0/model --batch_size=4 --eval_size=1000000 --configs defaults defaults_rlhf pythia_rlhf_individual

Replace `runs/ppo_individual/models_rm-pythia-44m_seed0/sqrt_chi2_in_loss-0.0008/seed_0/model` with the path to your trained model (it should look similar to this based on where the train script outputs by default).

This will produce a JSON file `gold_eval_with_response_ids.json` in the model directory. The JSON file contains a sampled response to each of the prompt's in Coste et al.'s test set. The responses are scored by both the proxy and true reward models. We report the mean proxy and true reward across all responses in our paper.

## Citation
If you use the code in this repository, please consider [citing our work](https://github.com/cassidylaidlaw/orpo#citation) and/or [Coste et al.'s](https://github.com/tlc4418/llm_optimization#citation).
