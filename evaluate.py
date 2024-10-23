from tqdm import tqdm
from argparse import ArgumentParser
import torch
import numpy as np

from loss import TotalLoss
from metrics import *
from utils import set_seed
from fast_depth import FastDepth
from dataloaders import get_torso_real_eval_loader, get_torso_sim_eval_loader, get_feldraum_eval_dataloader
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # evtl cuda:0


class Evaluator:
    def __init__(self, 
                 model, 
                 data_loader, 
                 standalone: bool = True, 
                 writer = None, 
                 display_images: bool = False, 
                 per_image: bool = False, 
                 ):
        self._model = model
        self._data_loader = data_loader
        self._standalone = standalone
        self._writer = writer
        self._display_images = display_images
        self._per_image = per_image

    def evaluate(self, epoch):
        self._model.eval()

        if not self._standalone:
            print(f"Validating Epoch {epoch}")

        total_loss = 0
        loss = TotalLoss()
        mae = MAE(len(self._data_loader))
        rmse = RMSE(len(self._data_loader))
        fop_25 = FracOfPix(len(self._data_loader), .25)
        fop_50 = FracOfPix(len(self._data_loader), .5)
        fop_75 = FracOfPix(len(self._data_loader), .75)
        fop_100 = FracOfPix(len(self._data_loader), 1)

        # run evaluation
        with torch.no_grad():
            for data in tqdm(self._data_loader):
                images = data["img"].to(DEVICE)
                masks = data["mask"].to(DEVICE)
                
#                if self._torso_real:    
#                    labels = torch.zeros(size=(images.shape[0], 1, images.shape[2], images.shape[3])).to(DEVICE)
#                else:
                labels = data["depth"].to(DEVICE)

                outputs, _ = self._model(images)

                total_loss += loss(outputs, labels, masks)
                mae.update(outputs, labels, masks)
                rmse.update(outputs, labels, masks)
                fop_25.update(outputs, labels, masks)
                fop_50.update(outputs, labels, masks)
                fop_75.update(outputs, labels, masks)
                fop_100.update(outputs, labels, masks)

                if self._display_images:
                    fig, ax = plt.subplots(1,3)
                    
                    ax[0].imshow((outputs * masks)[0].cpu().numpy().squeeze(), vmin=0, vmax=1)
                    ax[0].set_title("Prediction")
                    
                    ax[1].imshow(np.moveaxis(images[0].cpu().numpy().squeeze(), 0, 2)[...,::-1], vmin=0, vmax=1)
                    ax[1].set_title("Image")
                    
                    ax[2].imshow((labels * masks)[0].cpu().numpy().squeeze(), vmin=0, vmax=1)
                    ax[2].set_title("Label")
                    plt.show()

                    #np.save("file.npy", (outputs * masks)[0].cpu().numpy().squeeze())
                    #np.save("img.npy", np.moveaxis(images[0].cpu().numpy().squeeze(), 0, 2)[...,::-1])
                    #exit()
                    # fig = plt.figure(figsize=(15, 10))
                    # ax = plt.axes(projection="3d")

                    # img = np.moveaxis(images[0].cpu().numpy().squeeze(), 0, 2)[...,::-1]
                    # depth_vis = (outputs * masks)[0].cpu().numpy().squeeze()
                    # STEP = 5
                    # for x in range(0, img.shape[0], STEP):
                    #     for y in range(0, img.shape[1], STEP):
                    #         ax.scatter(
                    #             [depth_vis[x, y]] * 3,
                    #             [img.shape[1] - y] * 3,
                    #             [img.shape[0] - x] * 3,
                    #             c=tuple(img[x, y, :3] / 255),
                    #             s=3,
                    #         )
                    #     ax.view_init(45, 135)

                    # plt.show()

        if self._writer:
            self._writer.add_scalar(f"eval/loss", total_loss / len(self._data_loader), epoch)
            self._writer.add_scalar(f"eval/mae", mae.get(), epoch)
            self._writer.add_scalar(f"eval/rmse", rmse.get(), epoch)
            self._writer.add_scalar(f"eval/fop25", fop_25.get(), epoch)
            self._writer.add_scalar(f"eval/fop50", fop_50.get(), epoch)
            self._writer.add_scalar(f"eval/fop75", fop_75.get(), epoch)
            self._writer.add_scalar(f"eval/fop100", fop_100.get(), epoch)

        self._model.train()

    # Prepare return value for the case where in "training" mode
        r_dict = {
            "avg_loss": total_loss / len(self._data_loader),
            "mae": mae.get().item(),
            "rmse": rmse.get().item(),
            "fop25": fop_25.get().item(),
            "fop50": fop_50.get().item(),
            "fop75": fop_75.get().item(),
            "fop100": fop_100.get().item()
        }

        print(r_dict)#, total_loss, len(data_loader))

        if self._per_image:
            np.savez("per_image", 
                     mae=mae.get_per_image().cpu().numpy(), 
                     rmse=rmse.get_per_image().cpu().numpy(),
                     fop25=fop_25.get_per_image().cpu().numpy(),
                     fop50=fop_50.get_per_image().cpu().numpy(),
                     fop75=fop_75.get_per_image().cpu().numpy(),
                     fop100=fop_100.get_per_image().cpu().numpy()
                     )

        return r_dict


if __name__ == "__main__":
    INPUT_SHAPE = (256, 256)

    set_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, help="number of workers used during eval")
    parser.add_argument("--checkpoint", type=str, help="Which checkpoint to evaluate")
    parser.add_argument("--display_images", action="store_true")
    parser.add_argument("--per_image", action="store_true")
    parser.add_argument("--dataset", type=str, help="Name of dataset used for evaluation")
    args = parser.parse_args()

    model = FastDepth()
    model.load(args.checkpoint)
    model.to(DEVICE)

    config = {"eval": {"dset_size": 0}, "general": {"num_workers": args.num_workers}}
    if args.dataset == "torso_sim":
        data_loader = get_torso_sim_eval_loader(config, INPUT_SHAPE)
    elif args.dataset == "torso_real":
        data_loader = get_torso_real_eval_loader(config, INPUT_SHAPE)
    elif args.dataset == "feldraum":
        data_loader = get_feldraum_eval_dataloader(config, INPUT_SHAPE)

    evaluator = Evaluator(model, data_loader, True, None, args.display_images, args.per_image)
    evaluator.evaluate(0)
