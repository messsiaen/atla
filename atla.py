import os
import yaml
import random
import argparse
import torch as th
import numpy as np
import pandas as pd
import tempfile
from tensorboardX import SummaryWriter
from ray import tune
from ray.tune.schedulers.pb2 import PB2, PopulationBasedTraining
from ray.tune import Checkpoint, run, sample_from 

from models.model_registry import Model, Strategy
from environments.EVS import ENV_EVS
from utilities.util import convert, dict2str
from utilities.trainer import PGTrainer

def rl(config):

    alg_name ="ATLA"
    env_name ="EVS"

    env_args_file = f"./args/env_args/{env_name}.yaml"
    with open(env_args_file, "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict.get("data_path", "").split("/")
    env_config_dict["data_path"] = "/".join(data_path)

    with open("./args/default.yaml", "r", errors='ignore') as f:
        default_config_dict = yaml.safe_load(f)

    if env_name == "EVS":
        env = ENV_EVS.EVSEnv(env_config_dict)
        default_config_dict["continuous"] = True

    alg_args_file = f"./args/alg_args/{alg_name}.yaml"
    with open(alg_args_file, "r", errors='ignore') as f:
        alg_config_dict = yaml.safe_load(f)["alg_args"]

    log_name = "-".join([env_name, alg_name])
    alg_config_dict = {**default_config_dict, **alg_config_dict, **config}
    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()

    args = convert(alg_config_dict)
    constraint_model = None

    alpha = config["alpha"]
    beta = config["beta"]

    save_path = f"pb2_a{alpha}_b{beta}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = SummaryWriter(save_path + "tensorboard/" + log_name)

    model = Model[alg_name]
    strategy = Strategy[alg_name]
    print(f"Training parameters: {args}\n")

    trainer = PGTrainer(args, model, env, logger, constraint_model)

    log_file = save_path + "tensorboard/" + log_name + "/log.txt"
    with open(log_file, "w+") as file:
        alg_args_str = dict2str(alg_config_dict, 'alg_params')
        env_args_str = dict2str(env_config_dict, 'env_params')
        file.write(alg_args_str + "\n")
        file.write(env_args_str + "\n")

    config["wandb"] = False

    start_episode = 0
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            checkpoint_data = th.load(
                os.path.join(checkpoint_dir, "checkpoint.pt"),
                map_location=device
            )
            start_episode = checkpoint_data["episode"] + 1
            trainer.behaviour_net.load_state_dict(checkpoint_data["model_state_dict"])
            print(f"== resume from checkpoint, continue with epoch {start_episode} \n")

    rewards = []

    for i in range(start_episode, args.train_episodes_num):
        stat = {}
        hyperparams = {"alpha": alpha, "beta": beta, "seed": seed}
        trainer.run(stat, i, hyperparams=hyperparams)
        trainer.logging(stat, config["wandb"])
        current_reward = stat['mean_train_reward']

        print(f"Episode {i}, reward: {current_reward}")
        rewards.append(current_reward)

        if len(rewards) < 160:
            objective = np.var(rewards)
        else:
            objective = np.var(rewards[-160:])

        if i % 80 == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                th.save(
                    {
                        "episode": i,
                        "model_state_dict": trainer.behaviour_net.state_dict()
                    },
                    os.path.join(tempdir, "checkpoint.pt")
                )
                tune.report({"objective": objective}, checkpoint=Checkpoint.from_directory(tempdir))

    logger.close()

if __name__ == "__main__":
    
    pb2 = PB2(
        metric="objective",
        mode="min",
        quantile_fraction=0.25,
        perturbation_interval=160,
        hyperparam_bounds={
            "alpha": [0.05, 0.2],
            "beta": [0.3, 0.6]
        }
    )

    for seed in range(0, 4):
        analysis = run(
            rl,
            scheduler=pb2,
            num_samples=4,
            reuse_actors=True,
            config = {
                "alpha": sample_from(lambda spec: random.uniform(0.05, 0.2)),
                "beta": sample_from(lambda spec: random.uniform(0.4, 0.6)),
                "seed": seed
            }
        )

        all_dfs = analysis.trial_dataframes
        names = list(all_dfs.keys())

        results = pd.DataFrame()
        for i in range(4):
            df = all_dfs[names[i]].copy()
            df['sample_num'] = i 
            results = pd.concat([results, df]).reset_index(drop=True)

        dir = "{}_{}_{}_Size{}_{}_{}_{}_{}_{}".format(rl, "file", "method", str(4), "env", "default", "max", "160", "batch")
        exist_dir = os.path.expanduser('~/data/' + dir)
        if not(os.path.exists(exist_dir)):
            os.makedirs(exist_dir)

        result_dir1 = os.path.expanduser('~/data/')
        result_dir2 = "{}/seed{}.csv".format(dir, str(seed))
        results.to_csv(result_dir1 + "{}/seed{}.csv".format(dir, str(seed)))