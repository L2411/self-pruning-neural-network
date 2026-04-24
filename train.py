import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import PrunableNet, PrunableLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Model
model = PrunableNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

lambda_sparse = 1e-4  # sparsity strength


# Sparsity Loss
def sparsity_loss(model):
    loss = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            loss += gates.sum()
    return loss


# Training
for epoch in range(3):  # small for now
    model.train()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        loss1 = criterion(outputs, labels)
        loss2 = sparsity_loss(model)

        loss = loss1 + lambda_sparse * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")


# Testing
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, pred = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (pred == labels).sum().item()

accuracy = 100 * correct / total
print("Accuracy:", accuracy)

# Calculate Sparsity %
total = 0
pruned = 0

for layer in model.modules():
    if isinstance(layer, PrunableLinear):
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.numel()
        pruned += (gates < 1e-2).sum().item()

sparsity = 100 * pruned / total
print("Sparsity:", sparsity)
