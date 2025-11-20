import os
import argparse
import yaml
import pickle

import torch
import numpy as np
import random

from train import train
from test import test


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setting seed
SEED = 42

def main():
    set_seed(SEED)

    # Read config file by reading the arg 'config' from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Getting device
    device = "cpu"
    if torch.cuda.is_available() and config["DEVICE_ID"] is not None:
        device = torch.device(f"cuda:{config['DEVICE_ID'][0]}")

    test_only = config["TRAIN"].get("TEST_ONLY", False)
    if test_only:
        test(config, device=device)
    else:
        hp_tuning = config.get("HP_TUNING", False)
        if not hp_tuning:
            train(config, device)

        else:
            try:
                import optuna
            except ImportError:
                raise ImportError(
                    "Please install optuna to use hyperparameter tuning or set HP_TUNING to False"
                )

            n_trials = config["N_TRIALS"]

            sampler = optuna.samplers.CmaEsSampler(warn_independent_sampling=False)

            # Loading sampler
            if os.path.exists("tuning/sampler.pkl"):
                with open("tuning/sampler.pkl", "rb") as f:
                    sampler = pickle.load(f)

            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                load_if_exists=True,
                study_name="pawvit_study",
                storage="sqlite:///pawvit.db",
            )

            study.optimize(
                lambda trial: train(config, device, trial)[1], n_trials=n_trials
            )

            # Saving sampler
            with open("tuning/sampler.pkl", "wb") as f:
                pickle.dump(study.sampler, f)

            print("Number of finished trials: ", len(study.trials))
            print("Best parameters:")
            print(study.best_params)
            print("Best mmAP:")
            print(study.best_value)


if __name__ == "__main__":
    main()
