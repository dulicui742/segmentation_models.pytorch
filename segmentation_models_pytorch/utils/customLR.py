class Custom1:

    def __init__(self, *, lr, min_lr, optimizer, pre_loss, lr_decay): 
        self.optimizer = optimizer
        self.pre_loss = pre_loss
        self.lr_decay = lr_decay
        self.lr = lr
        self.min_lr = min_lr

    def step(self, loss_value):
        self.loss_value = loss_value
        if loss_value > self.pre_loss * 1.0:
            old_lr = self.lr
            self.lr = self.lr * self.lr_decay
            print(f"lr_decay called: from {old_lr} to {self.lr}")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        self.pre_loss = loss_value

    def should_break(self) -> bool:
        return self.lr < self.min_lr

    def __repr__(self):
        return f"""{self.__class__.__name__}(
            lr={self.lr},
            lr_decay={self.lr_decay},
            min_lr={self.min_lr},
            pre_loss={self.pre_loss},
            optimizer={self.optimizer},
            loss_meter={self.loss_value},
        )"""
