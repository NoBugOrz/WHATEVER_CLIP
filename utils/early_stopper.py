import torch
import numpy as np

class EarlyStopping:
    def __init__(
            self,
            patience=10,  # 容忍多少个epoch无改善
            min_improvement=1e-4,  # 指标至少改善多少才算有效
            monitor="val_loss",  # 监控指标："val_loss"或"val_acc"等
            mode="min",  # 指标优化方向："min"（越小越好）或"max"（越大越好）
            save_path="best_model.pth"  # 最佳模型保存路径
    ):
        self.patience = patience
        self.min_improvement = min_improvement
        self.monitor = monitor
        self.mode = mode
        self.save_path = save_path

        # 初始化内部状态
        self.counter = 0  # 记录无改善的epoch数
        self.best_score = None  # 最佳指标分数
        self.early_stop = False  # 是否触发早停
        self.best_model_params = None  # 最佳模型参数

    def __call__(self, current_score, model):
        """
        每轮验证后调用，检查是否需要早停
        参数：
            current_score: 当前epoch的监控指标值
            model: 当前模型（用于保存最佳参数）
        """
        # 初始化最佳分数
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(model)
            return

        # 判断当前指标是否改善
        if self.mode == "min":
            # 越小越好的指标（如loss）：当前值 < 最佳值 * (1 - min_improvement) 才算改善
            improvement = (self.best_score - current_score) / self.best_score
            if improvement > self.min_improvement:
                self.best_score = current_score
                self._save_best_model(model)
                self.counter = 0  # 重置计数器
            else:
                self.counter += 1  # 无改善，计数器+1
        else:
            # 越大越好的指标（如accuracy）：当前值 > 最佳值 * (1 + min_improvement) 才算改善
            improvement = (current_score - self.best_score) / self.best_score
            if improvement > self.min_improvement:
                self.best_score = current_score
                self._save_best_model(model)
                self.counter = 0
            else:
                self.counter += 1

        # 检查是否触发早停
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"\n早停触发！连续{self.patience}个epoch无有效改善")
            print(f"最佳{self.monitor}值: {self.best_score:.6f}")

    def _save_best_model(self, model):
        """保存最佳模型参数"""
        torch.save(model.state_dict(), self.save_path)
        # 可选：保存完整模型（占用更多空间）
        # torch.save(model, self.save_path)

    def load_best_model(self, model):
        """加载最佳模型参数到模型中"""
        model.load_state_dict(torch.load(self.save_path))
        return model