import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    """
    带热身的学习率调度器
    功能：
    1. 热身阶段（前warmup_epochs个epoch）：学习率从min_lr增长到target_lr
    2. 衰减阶段（热身结束后）：学习率从target_lr余弦退火至min_lr
    """

    def __init__(
            self,
            optimizer,
            warmup_epochs: int,  # 热身的epoch数量
            total_epochs: int,  # 总训练epoch数量
            target_lr: float,  # 热身结束时达到的目标学习率
            warmup_type: str = "cosine",  # 热身类型
            min_lr: float = 1e-6,  # 初始最小学习率
            last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.warmup_type = warmup_type.lower()
        self.min_lr = min_lr

        assert self.warmup_type in ["linear", "cosine"], f"不支持的热身类型: {warmup_type}"
        assert self.warmup_epochs <= self.total_epochs, "热身epoch不能超过总epoch"
        assert self.min_lr < self.target_lr, "min_lr必须小于target_lr"

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1  # 从1开始计数（第1个epoch）

        # 1. 热身阶段：学习率从min_lr增长到target_lr
        if current_epoch <= self.warmup_epochs:
            if self.warmup_type == "linear":
                # 线性增长：进度比例 * (目标-最小) + 最小
                progress = current_epoch / self.warmup_epochs
                lr = self.min_lr + (self.target_lr - self.min_lr) * progress
            else:  # cosine
                # 余弦增长：更平滑，初期增长慢，后期加速
                progress = current_epoch / self.warmup_epochs
                lr = self.min_lr + (self.target_lr - self.min_lr) * (1 - math.cos(progress * math.pi / 2))

        # 2. 衰减阶段：热身结束后，余弦退火至min_lr
        else:
            # 计算剩余epoch的进度（0~1）
            remaining_epochs = self.total_epochs - self.warmup_epochs
            progress = (current_epoch - self.warmup_epochs) / remaining_epochs
            # 余弦退火公式：从target_lr平滑衰减到min_lr
            lr = self.min_lr + (self.target_lr - self.min_lr) * (1 + math.cos(progress * math.pi)) / 2

        return [lr for _ in self.base_lrs]