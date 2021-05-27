"""
Run multiple experiments on a single machine.
"""
import subprocess

import numpy as np

ALGOS = ["sac", "msac"]
ENVS = ["AntBulletEnv-v0"]
N_SEEDS = 1
# EVAL_FREQ = 10000
# N_EVAL_EPISODES = 5
# LOG_STD_INIT = [-6, -5, -4, -3, -2, -1, 0, 1]

for seed in range(N_SEEDS):
    for algo in ALGOS:
        for env_id in ENVS:
        # for log_std_init in LOG_STD_INIT:
        #     log_folder = f"logs_std_{np.exp(log_std_init):.4f}"
            args = [
                "--algo",
                algo,
                "--env",
                env_id,
                # "--hyperparams",
                # f"policy_kwargs:dict(log_std_init={log_std_init}, net_arch=[64, 64])",
                # "--eval-episodes",
                # N_EVAL_EPISODES,
                # "--eval-freq",
                # EVAL_FREQ,
                # "-f",
                # log_folder,
                "--tensorboard-log",
                "logs/tensorboard",
                "--n-timesteps",
                1000000,
                "--seed",
                seed
            ]
            args = list(map(str, args))

            ok = subprocess.call(["python", "train.py"] + args)
