import torch
from torch import nn

from aff_resnet import resnet152, resnet18, resnet101, resnet34, resnet50
from fusion import MS_CAM


class SiameseNetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = resnet18(fuse_type='DAF', small_input=False)
        # self.encoder = MS_CAM()

        emb_len = 1000
        self.last = nn.Sequential(
            nn.Linear(4 * emb_len, 200, bias=False),
            nn.BatchNorm1d(200, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, input1, input2):
        emb1 = self.encoder(input1)
        emb2 = self.encoder(input2)

        x1 = torch.pow(emb1, 2) - torch.pow(emb2, 2)
        x2 = torch.pow(emb1 - emb2, 2)
        x3 = emb1 * emb2
        x4 = emb1 + emb2

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.last(x)

        return x
