  
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Qwen2MoeConfig():   
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        num_experts_per_tok=4,
        num_experts=60,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers


class Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size  # 1408
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts  # 60, 60个专家
        self.top_k = config.num_experts_per_tok  # 4, top_4
        self.norm_topk_prob = config.norm_topk_prob  # false

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape  # torch.Size([1, 128, 2048])
        hidden_states = hidden_states.view(-1, hidden_dim)             # torch.Size([128, 2048])
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)                       # torch.Size([128, 60]), 60是专家数量, Gate_Layer是一个线性层，用于选专家


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # torch.Size([128, 60]), 对每一行做softmax, 获得每个token对60个专家的得分
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # torch.Size([128, 4]), torch.Size([128, 4]), 每个token对应的top4专家的权值，和专家索引
        if self.norm_topk_prob:  # 如果为True, 则再做一次归一化
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)  # fp32 => fp16

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )  # torch.Size([128, 2048])

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  # torch.Size([60, 4, 128]), 每个专家是哪些token的top_x

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):  # 遍历60个专家
            expert_layer = self.experts[expert_idx]
            top_x, idx = torch.where(expert_mask[expert_idx])  # expert_mask[0]是[4,128],表示专家0对应哪些token(idx:[0~128, ...])的top_x([0~3, ...])，

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, idx].reshape(-1, hidden_dim)  # 选出当前专家对应的token, [11, 2048]表示当前专家对应11个token
            current_hidden_states = expert_layer(current_state) * routing_weights[idx, top_x, None]
            # current_hidden_states = expert_layer(current_state)
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `idx` tensor here.
            final_hidden_states.index_add_(0, idx, current_hidden_states.to(hidden_states.dtype))


        shared_expert_output = self.shared_expert(hidden_states)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output  # torch.Size([128, 2048])

        # final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


def get_shared_expert_weight(model):
    model = model.shared_expert

    weight_list  = []
    for param in model.parameters():
        weight_list.append(param)

    linear_weight1 = torch.cat(weight_list[0:2],dim=0).T

    linear_weight2 =weight_list[2].T

    return linear_weight1, linear_weight2


def get_gate_weight(model):
    model = model.gate
    weight_list  = []

    for param in model.parameters():
        param = F.pad(param, (0, 0, 0, 4)).detach()
        weight_list.append(param)
    

    weight = weight_list[0].T
    return weight

def get_expert_weight(model):
    linear_weight1=[]
    linear_weight2=[]

    model = model.experts

    for item in model:
        gate_proj_weight= item.gate_proj.weight.detach()
        up_proj_weight = item.up_proj.weight.detach()
        down_proj_weight = item.down_proj.weight.detach()

        linear_weight1.append((torch.cat([gate_proj_weight,up_proj_weight],dim=0).T)[None])
        linear_weight2.append((down_proj_weight.T)[None])

    expert_weight1 = torch.cat(linear_weight1,dim=0)
    expert_weight2 = torch.cat(linear_weight2,dim=0)

    return expert_weight1,expert_weight2


def get_shared_expert_gate_weight(model):
    model=model.shared_expert_gate
    weight = model.weight.detach().T


    return weight


def get_weight(model):

    get_gate_weight(model)
    get_expert_weight(model)
    get_shared_expert_weight(model)
    get_shared_expert_gate_weight(model)

import bisect

def find_total_elts_leq_target(sorted_indices, target):
    """
    使用二分查找找到小于等于目标专家编号的元素数量。
    """
    # bisect_right 返回排序列表中目标值应插入的位置
    # 该位置后的所有元素都大于目标值
    return bisect.bisect_right(sorted_indices, target)

def compute_total_rows_before_expert(sorted_experts, num_experts):
    """
    计算每个专家在 sorted_experts 中出现的总行数。
    """
    total_rows_before_expert = [0] * num_experts  # List, len=60
    for expert in range(num_experts):
        total_rows_before_expert[expert] = find_total_elts_leq_target(sorted_experts, expert)
    return np.array(total_rows_before_expert)

@torch.no_grad()
def expert_resetv2(topk_indices,token_expert_indices):
    
    topk_indices_flat = topk_indices.view(-1)  # 展平 [128,4] => [512]
    token_expert_indices_flat = token_expert_indices.view(-1) # 展平 [128,4] => [512]

    vas = torch.linspace(0,0.99,topk_indices_flat.shape[0])  # 产生512个从0到0.99等距分布的数字

    # Sort the indices based on topk_indices
    sorted_indices = torch.argsort(topk_indices_flat+vas)


    sorted_topk_indices = topk_indices_flat[sorted_indices]
    sorted_token_expert_indices = token_expert_indices_flat[sorted_indices]
    return sorted_topk_indices, sorted_token_expert_indices

@torch.no_grad()
def fused_softmax_topk(gating_output, k):
    seq_len = gating_output.shape[0]  # seq_len = 128
    routing_weights = F.softmax(gating_output, dim=1, dtype=torch.float)  # 计算每个token对应expert的得分，作为权值
    topk_weights, topk_indices = torch.topk(routing_weights, k, dim=-1)   # [128, 4], 选出每个token对应的top4的expert，返回对应的得分权值，以及专家索引
    token_expert_indices = (torch.arange(k)*seq_len).view(1,k).repeat(seq_len,1)+(torch.arange(seq_len).view(-1,1).repeat(1,4))  # [128, 4]

    topk_weights = topk_weights.to(gating_output.dtype)
    return topk_weights, topk_indices, token_expert_indices

def dict_sort(topk_indices,token_expert_indices):
    topk_indices_flat = topk_indices.view(-1)
    token_expert_indices_flat = token_expert_indices.view(-1)

    key_array = topk_indices_flat.numpy().tolist()
    value_array = token_expert_indices_flat.numpy().tolist()

    zipped_pairs = list(zip(key_array, value_array))

# 根据 keys 对打包的对进行排序
    sorted_pairs = sorted(zipped_pairs, key=lambda pair: pair[0])

    # 解压缩成两个独立的列表
    sorted_keys, sorted_values = zip(*sorted_pairs)

    # 如果需要列表形式，可以将结果再转换成列表
    sorted_keys = list(sorted_keys)
    sorted_values = list(sorted_values)

    return torch.tensor(sorted_keys),torch.tensor(sorted_values)

@torch.no_grad()
def expert_copy(inputs, sorted_token_expert_indices):
    seq_len, hidden_units = inputs.shape  # [128, 2048]
    copy_outputs = torch.zeros(seq_len*k, hidden_units).half()  # torch.Size([512, 2048]), 每个token要分配4个专家
    dst_2_src_line = torch.zeros(seq_len*k, dtype=torch.int32)  # torch.Size([512]), 记录每行对应
    for i in range(seq_len*k):
        dst_2_src_line[sorted_token_expert_indices[i]] = i      # 假设i=126, sorted_token_expert_indices[126] = 1, 那么dst_2_src_line[1] = 126, 说明，copy_outputs的第126行，对应第(1%128=1)个token，top (1/128+1 = 1)
        copy_outputs[i] = inputs[sorted_token_expert_indices[i] % seq_len]  # 复制4倍后的数组，copy_outputs是按照专家顺序排序的，比如copy_outputs[0:11]对应专家0,
    return copy_outputs, dst_2_src_line

@torch.no_grad()
def gated_silu(x, dimension):
    x1, x2 = x[:, :dimension], x[:, dimension:]
    x1 = x1 * torch.sigmoid(x1)
    return x1 * x2

@torch.no_grad()
def do_expertv2(copy_outputs,moe_weight1,moe_weight2,total_rows_before_expert):
    # copy_outputs.shape = torch.Size([512, 2048]), moe_weight1.shape = torch.Size([60, 2048, 2816]), moe_weight2.shape = torch.Size([60, 1408, 2048])
    gemm_outputs1 = torch.zeros(copy_outputs.shape[0],moe_weight1.shape[-1]).half()  # torch.Size([512, 1408*2])
    gemm_outputs2 = torch.zeros(copy_outputs.shape[0],moe_weight2.shape[-1]).half()  # torch.Size([512, 2048])


    pre_value=0
    for i in range(len(total_rows_before_expert)):  # [0, 60)
        if(i!=0):
            pre_value = total_rows_before_expert[i-1]
        
        token_length = total_rows_before_expert[i] - pre_value  # 根据前缀表计算当前专家处理的token数量

        if(token_length==0):
            continue

        lef_mat   = copy_outputs[pre_value:pre_value+token_length]  # 取出当前专家负责的那些token, 作为左矩阵, [11, 2048]
        right_mat = moe_weight1[i]  # 当前专家对应的linear层，包含gate_proj和up_proj, shape是[2048, 1408*2]

        gemm_outputs1[pre_value:pre_value+token_length] = lef_mat @ right_mat  # 进行Linear层的映射

    silu_outputs = gated_silu(gemm_outputs1, dimension=moe_weight1.shape[-1]//2)

    pre_value=0

    for i in range(len(total_rows_before_expert)):
        if(i!=0):
            pre_value= total_rows_before_expert[i-1]
        
        token_length = total_rows_before_expert[i] - pre_value

        if(token_length==0):
            continue

        lef_mat   = silu_outputs[pre_value:pre_value+token_length]
        right_mat = moe_weight2[i]

        gemm_outputs2[pre_value:pre_value+token_length] = lef_mat @ right_mat

    return gemm_outputs2

@torch.no_grad()
def moe_routing(gemm_outputs, topk_weights, dst_2_src_line):
    seq_len, hidden_units = gemm_outputs.shape  # torch.Size([512, 2048])
    seq_len = seq_len // 4  # 512 => 128

    outputs = torch.zeros(seq_len, hidden_units).half()  # torch.Size([128, 2048])
    # gemm_outputs是按照专家排序的，所以需要dst_2_src_line来根据token号，找到
    for i in range(seq_len):  # 遍历每个token
        for j in range(4):    # 遍历该token的每个专家
            outputs[i] += gemm_outputs[dst_2_src_line[j*seq_len+i]] * topk_weights[i][j]  # 

    return outputs

@torch.no_grad()
def ffn(final_outputs,weight1,weight2):
    x= final_outputs@weight1
    x = gated_silu(x,weight1.shape[-1]//2)
    x = x@weight2

    return x

@torch.no_grad()
def fused_sigmoid_dot_add(expert_output, shared_expert_output, shared_gate_output):
    shared_gate_output = torch.nn.functional.sigmoid(shared_gate_output)
    shared_expert_output = shared_expert_output * shared_gate_output
    outputs = expert_output + shared_expert_output

    return outputs

@torch.no_grad()
def mymoe_block(inputs,k,gate_mlp,m1,m2,s1,s2,gate):
    inputs = inputs.view(-1, inputs.shape[-1])  # [128, 2048]
    gate_outputs = gate_mlp(inputs)             # [128, 60]

    topk_weights, topk_indices, token_expert_indices = fused_softmax_topk(gate_outputs, k)
    sorted_topk_indices2, sorted_token_expert_indices2 = expert_resetv2(topk_indices, token_expert_indices)
    copy_outputs, dst_2_src_line = expert_copy(inputs, sorted_token_expert_indices2)
    total_rows_before_expert = compute_total_rows_before_expert(sorted_topk_indices2.tolist(), 60)
    moe_gemm_outputs = do_expertv2(copy_outputs, m1, m2, total_rows_before_expert)
    final_outputs = moe_routing(moe_gemm_outputs, topk_weights, dst_2_src_line)

    b = ffn(inputs, s1, s2)  # shared_expert
    c = gate(inputs)         # shared_gate_layer

    out = fused_sigmoid_dot_add(final_outputs, b, c)

    return out

if __name__ == "__main__":
    k = 4
    torch.manual_seed(14)
    config = Qwen2MoeConfig()
    inputs = torch.randn(1, 128, 2048).half()  # [bs, seq_len, hidden_dim]
    qwen_moe_block = Qwen2MoeSparseMoeBlock(config).eval().half()
    get_weight(qwen_moe_block)
    with torch.no_grad():
        m1, m2 = get_expert_weight(qwen_moe_block)  # [60, 2048, 1408*2], [60, 1408, 2048]
        s1, s2 = get_shared_expert_weight(qwen_moe_block)  # [2048, 5632*2], [5632, 2048]
        huggingface_outputs = qwen_moe_block(inputs)  # [seq_len, hidden_dim], [128, 2048]
        my_outputs = mymoe_block(inputs, k, qwen_moe_block.gate, m1, m2, s1, s2, qwen_moe_block.shared_expert_gate)

    print((huggingface_outputs - my_outputs).max())
    print("done")
