from typing import Dict
from omegaconf import DictConfig
import torch
import torch.nn as nn

from object_tracking.track.model.group_modules import GConv2d

class LinearPredictor(nn.Module):
    def __init__(self, x_dim: int, pix_dim: int):
        super().__init__()
        self.projection = GConv2d(x_dim, pix_dim + 1, kernel_size=1)

class AuxComputer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        use_sensory_aux = cfg.model.aux_loss.sensory.enabled
        self.use_query_aux = cfg.model.aux_loss.query.enabled

        sensory_dim = cfg.model.sensory_dim
        embed_dim = cfg.model.embed_dim

        if use_sensory_aux:
            self.sensory_aux = LinearPredictor(sensory_dim, embed_dim)
        else:
            self.sensory_aux = None