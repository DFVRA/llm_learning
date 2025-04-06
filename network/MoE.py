import torch
from torch import nn


class ExpertNetWork(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        output = self.linear2(x)
        return output


class Router(nn.Module):
    def __init__(self, hidden_size, expert_num, top_k):
        super().__init__()
        self.router = nn.Linear(hidden_size, expert_num)
        self.top_k = top_k
        self.hidden_size = hidden_size

    def forward(self, x):
        # (batch_size, seq_len, hidden_size) -> (batch_size*seq_len, hidden_size)
        x = x.view(-1, self.hidden_size)
        # (batch_size*seq_len, hidden_size)->(batch_size*seq_len, expert_num)
        x = self.router(x)
        x = nn.functional.softmax(x, dim=-1)
        topk_weight, topk_index = torch.topk(x, k=self.top_k, dim=-1, sorted=False)
        # topk 权重重新归一化输出
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        # (batch_size*seq_len, top_k)
        return topk_weight, topk_index


class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, expert_num, top_k):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_num = expert_num
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [ExpertNetWork(self.hidden_size, self.intermediate_size) for _ in range(expert_num)]
        )
        self.router = Router(self.hidden_size, self.expert_num, self.top_k)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        token_num = batch_size * seq_len
        x_flat = x.view(token_num, hidden_size)
        topk_weight, topk_index = self.router(x)
        # 初始化输出张量
        output = torch.zeros_like(x_flat)
        for token_idx in range(token_num):
            for expert_idx in range(self.top_k):
                expert = self.experts[topk_index[token_idx, expert_idx]]
                output[token_idx] += topk_weight[token_idx, expert_idx] * expert(x_flat[token_idx])
        output = output.view(batch_size, seq_len, hidden_size)
        return output


if __name__ == "__main__":
    hidden_size = 4096
    intermediate_size = 2048
    expert_num = 8
    top_k = 2
    inputs = torch.randn((2, 11, 4096))
    moe_layer = MoELayer(hidden_size, intermediate_size, expert_num, top_k)
    outputs = moe_layer(inputs)
    print(outputs.size())
