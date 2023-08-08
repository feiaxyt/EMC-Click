from torch.optim.lr_scheduler import _LRScheduler
import warnings
from collections import Counter
import numpy as np

class MultiStepLRWithWarmUp(_LRScheduler):
    def __init__(self, optimizer, min_lr, warmup_epoch, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.min_lr = min_lr
        self.warmup_epoch = warmup_epoch
        super(MultiStepLRWithWarmUp, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warmup_epoch:
            return [float(np.linspace(self.min_lr, base_lr, self.warmup_epoch)[self.last_epoch]) for base_lr in self.base_lrs]

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
