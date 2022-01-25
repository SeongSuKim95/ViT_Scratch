import torch


inp = torch.tensor([[0,4.],[-1,7],[3,5]]) #float
n_samples , n_features = inp.shape
module = torch.nn.LayerNorm(n_features, elementwise_affine = False)
# elementwise_affine False : No learnable parameters

print(sum(p.numel() for p in module.parameters() if p.requires_grad)) # No learnable parameters

print(inp.mean(-1),inp.std(-1,unbiased=False))

print(module(inp).mean(-1),module(inp).std(-1,unbiased =False)) # Layer norm make zero mean, 1 std

module = torch.nn.LayerNorm(n_features, elementwise_affine = True)

print(sum(p.numel() for p in module.parameters() if p.requires_grad)) # 4 Learnable parameters

print(module.bias, module.weight) # New per feature mean and standard deviation

print(module(inp).mean(-1),module(inp).std(-1,unbiased =False)) # Layer norm make zero mean, 1 std

module.bias.data +=1
module.weight.data *=4

print(module(inp).mean(-1),module(inp).std(-1,unbiased =False)) # Layer norm make zero mean, 1 std
