import optuna
import os
from train import Train
from utils import set_seed
from os.path import join

idx = 0

def objective(trial):
    global idx
    idx += 1

    weight_sim = trial.suggest_float("weight_sim", 0, 2, step=.1)
    weight_dis = trial.suggest_float("weight_dis", 0, 2, step=.1)
    weight_smo = trial.suggest_float("weight_smo", 0, 2, step=.1)

    config = {"general": {}, "train": {}, "eval": {}}
    config["general"] = {
        "name": f"trial_{idx}",
        "num_workers": 20,
        "training": True,
        "checkpoints_dir": join("/srv/ssd_nvm/21donn/checkpoints/depth_estimation", "optimisation"),
        "tensorboard_dir": join("/srv/ssd_nvm/21donn/tensorboard/depth_estimation", "optimisation", f"trial_{idx}")
    }
    config["train"] = {
        "weight_sim": weight_sim,
        "weight_dis": weight_dis,
        "weight_smo": weight_smo
    }
    config["eval"] = {
        "verbose": False,
    }

    # Make everything deterministic
    set_seed(42)
    trainer = Train(config)
    res = trainer.run(config)

    return res["mae"]


if __name__ == "__main__":
    name = __file__.split(".")[0].split(os.sep)[-1]
    #study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db', directions=["minimize" for i in range(6)], load_if_exists=True)
    study = optuna.create_study(study_name=name, storage=f'sqlite:///{name}.db', direction="minimize", load_if_exists=True)

    study.enqueue_trial({
        "weight_sim": 1,
        "weight_dis": 1,
        "weight_smo": 1
       })

    print(f"Sampler is {study.sampler.__class__.__name__}")

    study.optimize(objective, n_trials=50)


    