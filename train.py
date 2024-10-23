from argparse import ArgumentParser
from tqdm import tqdm
from os.path import join
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml

from evaluate import Evaluator
from loss import TotalLoss, ZoeDepthLoss
from dataloaders import *
from model import Model
from utils import set_seed
from best_model import *

BATCH_SIZE = 64
INPUT_SHAPE = (256, 256)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAXEPOCHS = 100


class Train:
    def __init__(self, config):
        # Data loaders
        dataset_train = config["train"]["dataset"]
        if dataset_train == "torso_sim":
            self._train_dataloader = get_torso_sim_train_loader(config, BATCH_SIZE, INPUT_SHAPE)
        elif dataset_train == "torso_real":
            self._train_dataloader = get_torso_real_train_loader(config, BATCH_SIZE, INPUT_SHAPE)
        elif dataset_train == "torso_sim_cycled":
            self._train_dataloader = get_torso_sim_cycled_train_loader(config, BATCH_SIZE, INPUT_SHAPE)
        elif dataset_train == "torso_sim_and_cycled":
            self._train_dataloader = get_torso_sim_and_cycled_train_loader(config, BATCH_SIZE, INPUT_SHAPE)

        dataset_test = config["eval"]["dataset"]
        if dataset_test == "torso_sim":
            self._test_dataloader = get_torso_sim_eval_loader(config, INPUT_SHAPE)
        elif dataset_test == "torso_real":
            self._test_dataloader = get_torso_real_eval_loader(config, INPUT_SHAPE)
        elif dataset_test == "torso_sim_cycled":
            self._test_dataloader = get_torso_sim_cycled_eval_loader(config, INPUT_SHAPE)
        elif dataset_test == "torso_sim_and_cycled":
            self._test_dataloader = get_torso_sim_and_cycled_eval_loader(config, INPUT_SHAPE)

        # Init network
        self._model = Model()
        self._model.to(device=DEVICE)

        # Loss
        #self._depth_loss = TotalLoss(
        #    weight_dis=config["train"]["weight_dis"], 
        #    weight_smooth=config["train"]["weight_smo"]
        #)
        
        self._depth_loss = ZoeDepthLoss("mean")
        
        # Optimization
        self._optimizer = optim.AdamW(self._model.parameters(), weight_decay=1e-8)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=.5, patience=10)

        self._writer = SummaryWriter(log_dir=config["general"]["tensorboard_dir"])

        self._evaluator = Evaluator(self._model, self._test_dataloader, False, self._writer)

    def run(self, config):
        """
        Run the entire network training pipeline for MAXEPOCHS
         """
        
        best_model_fop25 = BestModel("fop25", OptDirection.MAXIMIZE, join(config["general"]["checkpoints_dir"], "best_fop25"))
        best_model_rmse = BestModel("rmse", OptDirection.MINIMIZE, join(config["general"]["checkpoints_dir"], "best_rmse"))
        best_model_mae = BestModel("mae", OptDirection.MINIMIZE, join(config["general"]["checkpoints_dir"], "best_mae"))


        for epoch in range(1, MAXEPOCHS + 1):
            config["train"]["epoch"] = epoch
            self._writer.add_scalar("learning_rate", self._optimizer.param_groups[0]['lr'], epoch)

            # run a single epoch
            self._train_epoch(epoch=epoch)

            # evaluate after each epoch
            eval_dict = self._evaluator.evaluate(epoch)

            # LR scheduler step
            self._scheduler.step(eval_dict["avg_loss"])

            self._model.save_as(config["general"]["checkpoints_dir"], f"{config['general']['name']}_{epoch}")

            best_model_fop25.update(epoch=epoch, model=self._model, metric_value=eval_dict["fop25"])
            best_model_rmse.update(epoch=epoch, model=self._model, metric_value=eval_dict["rmse"])
            best_model_mae.update(epoch=epoch, model=self._model, metric_value=eval_dict["mae"])
        
        self._writer.close()

        return eval_dict


    def _train_epoch(self, epoch):
        """
        Train a single epoch.
        """

        self._model.train()

        print(f"Training Epoch {epoch}")

        # Run the epoch
        for i, data_sim in enumerate(tqdm(self._train_dataloader)):
            inputs = data_sim["img"].to(DEVICE)
            depths = data_sim["depth"].to(DEVICE)
            masks = data_sim["mask"].to(DEVICE)
            
            self._optimizer.zero_grad()

            depth_output, _ = self._model(inputs)

            depth_loss = self._depth_loss(depth_output[:BATCH_SIZE, ...], depths, masks)

            total_loss = depth_loss

            total_loss.backward()

            self._optimizer.step()

            self._writer.add_scalar("train/total_loss", total_loss, epoch * (i + 1))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of job")
    parser.add_argument("--num_workers", type=int, help="number of workers used during training")
    parser.add_argument("--checkpoints_dir", type=str, help="Where to store the checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, help="Where to store the tensorboard logs")
    parser.add_argument("--weight_sim", type=float, default=0, help="Weight of similarity loss")
    parser.add_argument("--weight_dis", type=float, default=1, help="Weight of distance loss")
    parser.add_argument("--weight_smo", type=float, default=1, help="Weight of smoothness loss")
    parser.add_argument("--dataset_train", type=str, help="Name of dataset used for training")
    parser.add_argument("--dataset_eval", type=str, help="Name of dataset used for eval")
    args = parser.parse_args()

    # Configure training according to arguments
    config = {"general": {}, "train": {}, "eval": {}}
    config["general"] = {
        "name": args.name,
        "num_workers": args.num_workers,
        "training": True,
        "checkpoints_dir": join(args.checkpoints_dir, args.name),
        "tensorboard_dir": join(args.tensorboard_dir, args.name)
    }
    config["train"] = {
        "dataset": args.dataset_train,
        "weight_dis": args.weight_dis,
        "weight_smo": args.weight_smo
    }
    config["eval"] = {
        "dataset": args.dataset_eval,
        "verbose": False,
    }

    os.makedirs(config["general"]["checkpoints_dir"], exist_ok=True)

    with open(os.path.join(config["general"]["checkpoints_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Make everything deterministic
    set_seed(42)
    trainer = Train(config)
    trainer.run(config)

