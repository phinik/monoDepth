from argparse import ArgumentParser
from tqdm import tqdm
from os.path import join
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from evaluate import Evaluator
from loss import TotalLoss
from dataloaders import *
from model import Model
from utils import set_seed
from metric import Metric

import matplotlib.pyplot as plt

BATCH_SIZE = 24
INPUT_SHAPE = (256, 256)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAXEPOCHS = 50


class Train:
    def __init__(self, config):
        # Data loaders
        self._train_dataloader_sim = get_torso_sim_train_loader(config, BATCH_SIZE, INPUT_SHAPE)
        self._train_dataloader_real = get_torso_real_train_loader(config, BATCH_SIZE, INPUT_SHAPE)
        self._test_dataloader_sim = get_torso_sim_eval_loader(config, INPUT_SHAPE)

        # Init network
        self._model = Model()
        self._model.to(device=DEVICE)

        # Loss
        self._depth_loss = TotalLoss(weight_sim=config["train"]["weight_sim"],
                                weight_dis=config["train"]["weight_dis"], 
                                weight_smooth=config["train"]["weight_smo"])
        
        self._domain_loss = torch.nn.BCELoss()
        
        # Optimization
        self._optimizer = optim.AdamW(self._model.parameters(), weight_decay=1e-8)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, factor=.5, patience=10)

        self._writer = SummaryWriter(log_dir=config["general"]["tensorboard_dir"])

        self._evaluator = Evaluator(self._model, self._test_dataloader_sim, False, self._writer)

    def run(self, config):
        """
        Run the entire network training pipeline for MAXEPOCHS
         """
        
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
        
        self._writer.close()

        return eval_dict


    def _train_epoch(self, epoch):
        """
        Train a single epoch.
        """

        self._model.train()

        print(f"Training Epoch {epoch}")

        domain_metric = Metric(2)
        real_iter = iter(self._train_dataloader_real)
        # Run the epoch
        for i, data_sim in enumerate(tqdm(self._train_dataloader_sim)):
            inputs_sim, depths_sim, masks_train, domains_sim = data_sim[0].to(DEVICE), data_sim[1].to(DEVICE), data_sim[2].to(DEVICE), data_sim[3].to(DEVICE)

            print(inputs_sim.cpu().numpy()[0, ...].shape)
            plt.imshow(np.transpose(inputs_sim.cpu().numpy()[0, ...], (1, 2, 0)))
            plt.show()
            
            


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of job")
    parser.add_argument("--num_workers", type=int, help="number of workers used during training")
    parser.add_argument("--checkpoints_dir", type=str, help="Where to store the checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, help="Where to store the tensorboard logs")
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
        "weight_sim": 1,
        "weight_dis": 1,
        "weight_smo": 1
    }
    config["eval"] = {
        "verbose": False,
    }

    #os.makedirs(config["general"]["checkpoints_dir"], exist_ok=True)

    # Make everything deterministic
    set_seed(42)
    trainer = Train(config)
    trainer.run(config)

