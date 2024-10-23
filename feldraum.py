from metrics import *
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import join, splitext
import torch

zeo_base_path = "/srv/ssd_nvm/21donn/datasets/zoe_depth_kinect"
feldraum_base_path = "/srv/ssd_nvm/17vahl/feldraum_depth"
dirs = ["train", "test"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # evtl cuda:0

for dir in dirs:
    print(f"Comparing {dir}")
    files = listdir(join(zeo_base_path, dir))

    mae = MAE(len(files))
    rmse = RMSE(len(files))
    fop_25 = FracOfPix(len(files), .25)
    fop_50 = FracOfPix(len(files), .5)
    fop_75 = FracOfPix(len(files), .75)
    fop_100 = FracOfPix(len(files), 1)

    for file in tqdm(files):
        filename = splitext(file)[0]
        target = np.load(join(zeo_base_path, dir, file))
        
        with np.load(join(feldraum_base_path, dir, "depth", filename + "_depth_raw.npz")) as f:
            zoe = f["arr_0"]
        zoe = np.rot90(zoe, k=3)
        zoe = np.fliplr(zoe)

        zoe = zoe - np.min(zoe) + .75

        mask = zoe > 0.001

        zoe = torch.from_numpy(zoe).to(DEVICE)
        target = torch.from_numpy(target).to(DEVICE)
        mask = torch.from_numpy(mask).to(DEVICE)

        mae.update(zoe, target, mask)
        rmse.update(zoe, target, mask)
        fop_25.update(zoe, target, mask)
        fop_50.update(zoe, target, mask)
        fop_75.update(zoe, target, mask)
        fop_100.update(zoe, target, mask)

    r_dict = {
            "mae": mae.get(),
            "rmse": rmse.get(),
            "fop25": fop_25.get(),
            "fop50": fop_50.get(),
            "fop75": fop_75.get(),
            "fop100": fop_100.get()
    }

    print(r_dict)