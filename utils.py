import torch
from model import PrunableLinear


def compute_sparsity_loss(model):
    loss = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            loss += gates.sum()
    return loss


def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)

            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total
