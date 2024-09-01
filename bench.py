# Copyright (c) 2024, Tongkai Xu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.cpp_extension import load
from qwen_moe_block import *

# load build/CUDA_MOE.so
import sys
sys.path.append("build")
import CUDA_MOE


# ====== Qwen ======
torch.manual_seed(14)
inputs = torch.randn(1, 128, 2048)  # [bs, seq_len, hidden_dim]
config = Qwen2MoeConfig()
qwen_moe_block = Qwen2MoeSparseMoeBlock(config).eval()

print('=== profiling qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    huggingface_outputs = qwen_moe_block(inputs)  # [seq_len, hidden_dim], [128, 2048]
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


# ====== manual Python Qwen (CUDA based process) ======
k = 4                                       # top4
inputs = inputs.view(-1, inputs.shape[-1])  # [128, 2048]
gate_outputs = qwen_moe_block.gate(inputs)  # [128, 64]
m1_T, m2_T = get_expert_weight_T(qwen_moe_block)         # [60, 2048, 1408*2], [60, 1408, 2048]
s1_T, s2_T = get_shared_expert_weight_T(qwen_moe_block)  # [2048, 5632*2], [5632, 2048]

print('=== profiling manual qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    my_outputs = mymoe_block(inputs, gate_outputs, k, m1_T, m2_T, s1_T, s2_T, qwen_moe_block.shared_expert_gate, False)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print((huggingface_outputs - my_outputs).max())

# ====== manual CUDA Qwen ======
m1, m2 = get_expert_weight(qwen_moe_block)
print('=== profiling cuda qwen moe block ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    moe_result = CUDA_MOE.forward(inputs, gate_outputs, k, m1, m2)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

moe_result = moe_result.cpu()
print(f"max_diff = {(moe_result - my_outputs).max()}")
print(f"python output:\nsize({my_outputs.shape})\n{my_outputs}\n")
print(f"cuda output:\nsize({moe_result.shape})\n{moe_result}\n")