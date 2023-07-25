import numpy as np
import torch
import torch.nn as nn

from . import base
from . import functional as F
from ..base.modules import Activation
from ..losses.focal import FocalLoss as FL
from ..losses.dice import DiceLoss as DL
from ..losses.unified_focal_loss_pytorch import(
    SymmetricUnifiedFocalLoss as SUFL,
    AsymmetricUnifiedFocalLoss as AUFL,
    AsymmetricFocalTverskyLoss as AFTL,
    SymmetricFocalTverskyLoss as SFTL,
    AsymmetricFocalLoss as AFL,
    SymmetricFocalLoss as SFL,
)
from ..losses.tversky import(
    TverskyLoss as TL,
    TverskyLossFocal as TLF,
)


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss1(DL, base.Loss):
    pass


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    # pass
    def forward(self, y_pr, y_gt):
        # # torch.nn.functional.binary_cross_entropy 
        # # and torch.nn.BCELoss are unsafe to autocast.
        # m = nn.Sigmoid()
        # loss = nn.BCELoss()
        # bce_loss = loss(m(y_pr), y_gt) #torch.autograd.Variable(y_gt)
        # return bce_loss

        loss = nn.BCEWithLogitsLoss()
        bce_loss = loss(y_pr, y_gt)
        return bce_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class FocalLoss(FL, base.Loss):
    pass


class SymmetricUnifiedFocalLoss(SUFL, base.Loss):
    pass


class AsymmetricUnifiedFocalLoss(AUFL, base.Loss):
    pass


class AsymmetricFocalTverskyLoss(AFTL, base.Loss):
    pass


class SymmetricFocalTverskyLoss(SFTL, base.Loss):
    pass


class AsymmetricFocalLoss(AFL, base.Loss):
    pass


class SymmetricFocalLoss(SFL, base.Loss):
    pass


class TverskyLoss(TL, base.Loss):
    pass


class TverskyLossFocal(TLF, base.Loss):
    pass