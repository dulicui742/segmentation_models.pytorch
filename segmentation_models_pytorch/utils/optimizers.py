from itertools import chain
import torch
import torchvision as tv
import numpy as np
# from .yellowfin import YFOptimizer

# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
def get_adam_optimizer(model, lr, weight_decay=1e-5):
    parameters = model.parameters()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, parameters),
        lr=lr,
        weight_decay=weight_decay,
        # betas=(0.9, 0.999),
        betas=(0.9, 0.99), # same with keras
    )
    return optimizer


# def get_yellow(model, lr):
#     return YFOptimizer(
#         filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-4
#     )


def get_sgd_optimizer(model, lr, momentum, weight_decay=1e-5):
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return optimizer


def get_rms_optimizer(model, lr, alpha=0.9, momentum=0.9, weight_decay=1e-5, eps=1e-08):
    print(f"lr: {lr}, momentum: {momentum}, weight_decay: {weight_decay}, eps: {eps}")
    optimizer = torch.optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        alpha=alpha, #0.99, 
        eps=eps, 
        weight_decay=weight_decay, #0.9
        momentum=momentum,  #0.9
        centered=False
    )
    return optimizer