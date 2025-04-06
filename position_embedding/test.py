import torch
Q = torch.randn(4, 3)
K = torch.randn(4, 3)
V = torch.randn(4, 3)
print(torch.matmul(torch.matmul(Q, K.T), V))
Q[[0, 1]] = Q[[1, 0]]
K[[0, 1]] = K[[1, 0]]
V[[0, 1]] = V[[1, 0]]
print(torch.matmul(torch.matmul(Q, K.T), V))