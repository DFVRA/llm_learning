import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        """
        :param d_model: 输入特征维度为 512
        :param num_heads: 注意力头的数量为 8
        :param dropout: Dropout概率为 0.1
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisiable bu num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.tensor([self.head_dim ** 0.5], dtype=torch.float32))


    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size, seq_len, d_model]
        :param key: [batch_size, seq_len, d_model]
        :param value: [batch_size, seq_len, d_model]
        :param mask: 形状与 sttn_scores 相同或可以广播到 attn_scores 相同形状，掩盖 padding或未来信息
        :return:
            output: [batch_size, seq_len, d_model]
        """
        # Q K V: [batch_size, seq_len, d_model]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 重塑为多头形式
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-1e20"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)

        output = self.fc_out(output)

        return output, attn_weights


if __name__ == "__main__":
    """
    多头注意力机制的参数组成：d_model * d_model * 4 + d_model * 4
    attn_weights shape : [batch_size, num_heads, seq_len, seq_len]
    """
    Q = K = V = torch.rand(2, 5, 512)
    mask = torch.tril(torch.ones(5, 5))
    print(f"mask : {mask}")
    mha = MultiHeadAttention()
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"total parameters : {total_params}")
    output, attn_weights = mha(Q, K, V, mask)
    print(f"output shape : {output.shape}")
    print(f"attn weights : {attn_weights.shape}")































