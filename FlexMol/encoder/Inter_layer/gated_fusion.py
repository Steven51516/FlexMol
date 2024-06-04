import torch
import torch.nn as nn
from .base import InteractionLayer


class GatedFusionLayer(InteractionLayer):
    def __init__(self, v_dim, q_dim, output_dim=128, dropout_rate=0.1):
        super(GatedFusionLayer, self).__init__()
        self.v_transform = nn.Linear(v_dim, output_dim)
        self.q_transform = nn.Linear(q_dim, output_dim)
        self.gate_transform = nn.Linear(output_dim*2, output_dim)
        self.activation = nn.Tanh()
        self.output_dim = output_dim

    def get_output_shape(self):
        return self.output_dim

    def forward(self, v, q):
        v_proj = self.activation(self.v_transform(v))
        q_proj = self.activation(self.q_transform(q))

        concat_proj = torch.cat([v_proj, q_proj], dim=1)
        gate = torch.sigmoid(self.gate_transform(concat_proj))

        gated_output = gate * v_proj + (1 - gate) * q_proj
        return gated_output
