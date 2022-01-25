import torch

p = 0.5

# Training

module = torch.nn.Dropout(p)

print(module.training)

inp = torch.ones(3,5)

print(module(inp))
print(module(inp))
print(module(inp))

# Randomly erase 50%
# Multiply the remaining elements with the constant : 1/(1-p)

# Evaluation
# Dropout layer behaves like an identity mapping
module.eval()
print(module(inp))
print(module(inp))
print(module(inp))