import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed random number generators in order to provide deterministic
    behavior.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # new
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True