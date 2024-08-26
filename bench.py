import torch
from torch.utils.cpp_extension import load

from qwen_moe_block import *

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"

moe = load(name='moe', sources=['main.cpp', 'moe_kernels.cu'], extra_cuda_cflags=['-O2'])

# ====== Qwen ======
torch.manual_seed(14)
inputs = torch.randn(1, 128, 2048)  # [bs, seq_len, hidden_dim]
config = Qwen2MoeConfig()
qwen_moe_block = Qwen2MoeSparseMoeBlock(config).eval()

print('=== profiling qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    huggingface_outputs = qwen_moe_block(inputs)  # [seq_len, hidden_dim], [128, 2048]
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


# ====== manual python Qwen(based on logic of CUDA) ======
inputs = inputs.view(-1, inputs.shape[-1])  # [128, 2048]
gate_outputs = qwen_moe_block.gate(inputs)  # [128, 64]
k = 4  # top4
m1, m2 = get_expert_weight(qwen_moe_block)         # [60, 2048, 1408*2], [60, 1408, 2048]
s1, s2 = get_shared_expert_weight(qwen_moe_block)  # [2048, 5632*2], [5632, 2048]

print('=== profiling manual qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    my_outputs = mymoe_block(inputs, gate_outputs, k, m1, m2, s1, s2, qwen_moe_block.shared_expert_gate)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print((huggingface_outputs - my_outputs).max())


# ====== manual cuda Qwen ======
print('=== profiling cuda qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    moe_result = moe.forward(inputs, gate_outputs, k, m1, m2, s1, s2)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print("done")