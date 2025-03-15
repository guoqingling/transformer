import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 4
nhead = 2
dim_feedforward = 8
batch_size = 2
seq_len = 3

assert d_model % nhead == 0

# Encoder
print('Use nn.TransformerEncoderLayer:')
encoder_input = torch.randn(seq_len, batch_size, d_model)
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=0.0)
memory = encoder_layer(encoder_input)

print('Output:\n', memory)
print('Shape of input:', encoder_input.shape, 'Shape of output:', memory.shape)


# Encoder manual
print('\nUse layer by layer:')
# Input
X = encoder_input
X_flat = X.contiguous().view(-1, d_model) # [seq_len * batch_size, d_model] -> [6, 4]


# Multi-head attention layer
self_attn = encoder_layer.self_attn
W_in = self_attn.in_proj_weight
b_in = self_attn.in_proj_bias
W_out = self_attn.out_proj.weight
b_out = self_attn.out_proj.bias

QKV = F.linear(X_flat, W_in, b_in) # [seq_len * batch_size, d_model] -> [seq_len * batch_size, 3 * d_model]

Q, K, V = QKV.split(d_model, dim=1) # or use Q, K, V = torch.chunk(QKV, 3, dim=-1)
head_dim = d_model // nhead

def reshape_for_heads(x):
    return x.contiguous().view(seq_len, batch_size, nhead, head_dim).permute(1, 2, 0, 3).reshape(nhead * batch_size, seq_len, head_dim) # reshape for heads

Q = reshape_for_heads(Q)
K = reshape_for_heads(K)
V = reshape_for_heads(V)

scores = torch.bmm(Q, K.transpose(1, 2) / (head_dim ** 0.5))
attn_weight = F.softmax(scores, dim=-1)
attn_output = torch.bmm(attn_weight, V)

attn_output = attn_output.view(batch_size, nhead, seq_len, head_dim).permute(2, 0, 1, 3).contiguous()
attn_output = attn_output.view(seq_len, batch_size, d_model)

attn_output = F.linear(attn_output.view(-1, d_model), W_out, b_out)
attn_output = attn_output.view(seq_len, batch_size, d_model)


# Residual Connections and Norm
norm1 = encoder_layer.norm1
residual = X + attn_output # [seq_len, batch_size, d_model]
normalizated = F.layer_norm(residual, (d_model,), weight=norm1.weight, bias=norm1.bias) # [seq_len, batch_size, d_model]


# Feed forward layer
W_1 = encoder_layer.linear1.weight
b_1 = encoder_layer.linear1.bias
W_2 = encoder_layer.linear2.weight
b_2 = encoder_layer.linear2.bias
norm2 = encoder_layer.norm2

ffn_output = F.linear(normalizated.view(-1, d_model), W_1, b_1)
ffn_output = F.relu(ffn_output)
ffn_output = F.linear(ffn_output, W_2, b_2)
ffn_output = ffn_output.view(seq_len, batch_size, d_model)

residual2 = normalizated + ffn_output
normalizated2 = F.layer_norm(residual2, (d_model,), weight=norm2.weight, bias=norm2.bias)
print('Output:\n', normalizated2,'\nShape of output:', normalizated2.shape)


# compare
print('IS TURE? ',torch.allclose(normalizated2, memory))

'''
you should know:
permute, transpose, view/reshape, flatten function: https://developer.baidu.com/article/details/2795458
multiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
layer_norm: https://zhuanlan.zhihu.com/p/18446035638
source code: https://github.com/chunhuizhang/bert_t5_gpt/blob/main/tutorials/encoder-decoder/transformer-encoder-decoder.ipynb
'''