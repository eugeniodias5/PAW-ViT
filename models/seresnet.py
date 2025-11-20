# Copied from https://github.com/moskomule/senet.pytorch/tree/master
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet

from models.helpers import load_checkpoint
from models.senet import seresnet50


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_in"
        )  # For old pytorch, you may use kaiming_normal.
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class SeResNet50(nn.Module):
    def __init__(
        self,
        num_parts,
        layers=None,
        num_classes=1000,
        pretrained=True,
        use_direction=False,
        use_bnneck=False,
        num_directions=1,
        stride=1,
    ):
        super(SeResNet50, self).__init__()
        self.num_parts = num_parts

        if isinstance(pretrained, bool):
            self.model = seresnet50(pretrained=pretrained, layers=layers)
            print(f"Loading pretrained SE-Resnet50")
        else:
            self.model = seresnet50(pretrained=False, layers=layers)
            load_checkpoint(self.model, pretrained)

        if stride == 1:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
        )
        self.cls.apply(weights_init_kaiming)

        if num_classes:
            self.fc = nn.Linear(512, num_classes)
            self.fc.apply(weights_init_classifier)

        self.fc_direction = None
        if use_direction:
            # Create a binary classifier for the direction
            self.fc_direction = nn.Linear(512, num_directions)
            self.fc_direction.apply(weights_init_classifier)

        self.bnneck = None
        if use_bnneck:
            self.bnneck = nn.BatchNorm1d(512)
            self.bnneck.bias.requires_grad_(False)
            self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        y = self.model.layer0(x)
        y = self.model.layer1(y)
        y = self.model.layer2(y)
        y = self.model.layer3(y)
        y = self.model.layer4(y)

        y_avg = self.avgpool(y)
        y_avg = y_avg.view(y_avg.size(0), -1)
        y_avg = self.cls(y_avg)

        if self.bnneck:
            y_avg = self.bnneck(y_avg)

        logits = self.fc(y_avg) if hasattr(self, "fc") else None
        direction = (
            self.fc_direction(y_avg).squeeze()
            if hasattr(self, "fc_direction")
            else None
        )

        return y_avg, logits, direction.squeeze()

    def infer(self, x):
        # Return the flattened features
        y = self.model.layer0(x)
        y = self.model.layer1(y)
        y = self.model.layer2(y)
        y = self.model.layer3(y)
        y = self.model.layer4(y)

        y_avg = self.avgpool(y)
        y_avg = y_avg.view(y_avg.size(0), -1)

        return y_avg
