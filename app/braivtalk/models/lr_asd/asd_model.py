"""
LR-ASD model architecture and loss/scoring modules.

Vendored from:
  - https://github.com/Junhua-Liao/LR-ASD/blob/main/model/Model.py
  - https://github.com/Junhua-Liao/LR-ASD/blob/main/loss.py
  - https://github.com/Junhua-Liao/LR-ASD/blob/main/ASD.py

Paper: "LR-ASD: Lightweight and Robust Network for Active Speaker Detection" (IJCV 2025)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import Fusion, Detector
from .encoder import visual_encoder, audio_encoder


class ASD_Model(nn.Module):
    """Core audio-visual model for active speaker detection."""

    def __init__(self):
        super(ASD_Model, self).__init__()

        self.visualEncoder = visual_encoder()
        self.audioEncoder = audio_encoder()
        self.fusion = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.detector(x)
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self, x):
        x = torch.reshape(x, (-1, 128))
        return x


class lossAV(nn.Module):
    """Audio-visual loss module. Also contains the FC scoring head used at inference."""

    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels=None, r=1):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            # Inference mode: return raw speaking scores
            predScore = x[:, 1]
            predScore = predScore.t()
            predScore = predScore.view(-1).detach().cpu().numpy()
            return predScore
        else:
            x1 = x / r
            x1 = F.softmax(x1, dim=-1)[:, 1]
            nloss = self.criterion(x1, labels.float())
            predScore = F.softmax(x, dim=-1)
            predLabel = torch.round(F.softmax(x, dim=-1))[:, 1]
            correctNum = (predLabel == labels).sum().float()
            return nloss, predScore, predLabel, correctNum


class ASD(nn.Module):
    """
    Full ASD wrapper with model + scoring head.

    This class mirrors the original LR-ASD ASD class so that pretrained
    weights can be loaded via ``loadParameters()``.
    """

    def __init__(self):
        super(ASD, self).__init__()
        self.model = ASD_Model()
        self.lossAV = lossAV()

    def loadParameters(self, path, device="cpu"):
        """Load pretrained weights, handling 'module.' prefix from DataParallel."""
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=device, weights_only=False)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    # lossV weights are training-only; skip silently
                    if not origName.startswith("lossV."):
                        print("%s is not in the model." % origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write(
                    "Wrong parameter length: %s, model: %s, loaded: %s"
                    % (origName, selfState[name].size(), loadedState[origName].size())
                )
                continue
            selfState[name].copy_(param)
