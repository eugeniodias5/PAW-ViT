"""
Adapted from : https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation/blob/master/loss/__init__.py#L65
"""

import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_per_class(probs, target, fg_mask, eps=1e-6):
    probs = probs * fg_mask
    target = target * fg_mask

    num = 2.0 * torch.sum(probs * target, dim=(-2, -1))
    denom = torch.sum(probs + target, dim=(-2, -1))
    return (num + eps) / (denom + eps)


class ExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss and Cross Entropy
    Loss

    Original paper: https://arxiv.org/pdf/1809.00076.pdf

    See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight

    Note:
        Input for CrossEntropyLoss are the logits - Raw output from the model
    """

    def __init__(
        self,
        w_dice=0.5,
        w_cross=0.5,
        gamma=0.3,
        use_softmax=True,
        class_weights=None,
        filter=False,
    ):
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
        self.gamma = gamma
        self.use_softmax = use_softmax
        self.class_weights = class_weights
        self.filter = filter

    def forward(self, output, target, epsilon=1e-6, bg_thr=0.05):
        """
        logits: raw network scores
        target: Foreground classes
        """
        B, C, H, W = output.shape
        output = output.float()
        target = target.float()

        if self.use_softmax:
            probs = F.softmax(output, dim=1)
        else:
            probs = output

        fg_mask = (target.sum(1, keepdim=True) > bg_thr).float()
        if not self.filter:
            fg_mask = torch.ones_like(fg_mask).float()

        # Dice Loss
        dice_per_class = soft_dice_per_class(probs, target, fg_mask, eps=epsilon)
        dice_loss = torch.pow(
            -torch.log(dice_per_class.clamp_min(1e-8)), self.gamma
        ).mean()

        with torch.no_grad():
            if self.class_weights is None:
                freq = target.view(B, C, -1).sum(dim=(0, 2)) + epsilon
                label_w = torch.sqrt(freq.sum() / freq)
            else:
                label_w = self.class_weights.to(output.device).float()

        # CE loss
        output = output.clamp_min(1e-8)

        logp = (-torch.log(probs)).clamp_min(1e-8)

        ce = target * torch.pow(logp, self.gamma)
        ce = ce * label_w.view(1, C, 1, 1)
        ce = ce.sum(dim=1)
        ce = ce * fg_mask.squeeze(1)
        ce_loss = ce.mean()

        return self.w_dice * dice_loss + self.w_cross * ce_loss
