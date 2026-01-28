from torch import nn
from transformers import AutoModelForSequenceClassification
from functools import partial

import torch

class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
# A is initialized with random normal values scaled by std_dev
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim,rank)*std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, tensor):
         tensor = self.alpha * (tensor @ self.A @ self.B)
         return tensor

class LinearWithLora(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.alpha = alpha
        self.lora = LoraLayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, tensor):
        return self.linear(tensor) + self.lora(tensor)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
for param in model.parameters():
    param.requires_grad = False

print(model)



lora_rank = 8
lora_alpha = 16
lora_dropout = 0.5
lora_query= True
lora_value = True
lora_key = False
lora_mlp = False
lora_projection = False
lora_head = False

layers = []

assign_lora = partial(LinearWithLora, rank=lora_rank, alpha=lora_alpha)  # Freeze Rank and Alpha params

for layer in model.distilbert.transformer.layer:
    layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    layer.attention.v_lin = assign_lora(layer.attention.v_lin)

print(model)