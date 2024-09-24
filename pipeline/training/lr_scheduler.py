from typing import Callable

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from yacs.config import CfgNode as CN


# copied from https://github.com/facebookresearch/mae/blob/main/util/lars.py
# as is
class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """

    def __init__(
        self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1:
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["trust_coefficient"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])


class CosineAnnealingWarmRestartsWithPlateau(CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        patience=10,
        threshold=1e-4,
    ):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        self.patience = patience
        self.threshold = threshold
        self.num_bad_epochs = 0
        self.best = None

    def step(self, epoch=None, metric=None):
        if metric is not None:
            if self.best is None or metric < self.best - self.threshold:
                self.best = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self.T_cur = 0
                self.T_i = self.T_0
                self.num_bad_epochs = 0

        super().step(epoch=epoch)


def build_scheduler(
    scheduler_name: str,
    optimizer: Callable,
    lr_config: "CN",
):

    lr_scheduler = None
    if scheduler_name == "cosine_with_plateau":
        lr_scheduler = CosineAnnealingWarmRestartsWithPlateau(
            optimizer=optimizer,
            T_0=lr_config.decay_steps,
            T_mult=lr_config.t_mult,
            eta_min=lr_config.eta_min,
            patience=lr_config.patience,
            threshold=lr_config.threshold,
        )
    elif scheduler_name == "cosine":
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=lr_config.decay_steps,
            T_mult=lr_config.t_mult,
            eta_min=lr_config.eta_min,
        )
    elif scheduler_name == "lars":
        lr_scheduler = LARS(
            optimizer,
            trust_coef=lr_config.trust_coef,
            trust_clip=lr_config.trust_clip,
            eps=lr_config.eps,
        )

    return lr_scheduler
