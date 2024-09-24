import pytorch_lightning as pl
import numpy as np
import random
import torch


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    random.seed(seed)

    pl.seed_everything(seed, workers=True)


# TODO - ADD WandB
