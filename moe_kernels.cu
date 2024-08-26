// ref: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.8.0/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
#include <torch/types.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32                        // 每个warp的中线程的数量
#define MAX_BYTES_PER_LDG 16                // 向量化访存最大字节数(16字节, float4)
#define CEIL(a, b) ((a + b - 1) / (b))


// topkGatingSoftmax<<<16, (32,4)>>>
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__global__ void topkGatingSoftmax(const float* input,
                                  const int num_rows,
                                  const int k,
                                  float* output, 
                                  int* indices,
                                  int* source_rows)
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float); // 16/4=4, 每次从Global Mem中向量化读取的float数, float4
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;  // 64
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;  // 64/4=16, 一行16个线程
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;  // 4/4=1, 每个线程向量化读几次
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;  // 32*4=128, 一个warp可以读128个数据
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;  // 128/64=2, 一个warp处理的行数
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;  // 4*2=8,一个block处理多少行

    // 逐个维度计算, 计算线程的行偏移量 block ==> warp ==> thread_row
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;  // blockIdx.x(0~15), block的行偏移量, 相当于一个block去处理4*2=8行
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;  // warp偏移量, 因为一个warp处理2行, threadIdx.y(0~3), ROWS_PER_WARP=2
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;  // (0-31)/16=0 or 1, 当前线程处理第1行还是第2行
    const int thread_row = warp_base_row + thread_row_in_warp;  // 最终得到当前线程处理哪一行
    
    if (thread_row >= num_rows) return;

    // 计算出当前线程的列偏移量
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;  // input偏移到当前行
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;  // 0-32%16, [0,1,...,15]
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;  // [0,4,...,60]
    const float* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;  // 偏移到当前列

    float row_chunk[VPT];  // row_chunk[4]
    float4* row_chunk_vec_ptr = reinterpret_cast<float4*>(&row_chunk);
    const float4* vec_thread_read_ptr = reinterpret_cast<const float4*>(thread_read_ptr);

    // 向量化数据搬运, global ==> register
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)  
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // ===================== Softmax ========================
    // 需要按行求max和sum, 因为每个线程处理一个float4, 所以当前线程先各自求4个
    // 元素的max和sum, 然后用warp shuffle求一行的max和sum, 最终计算出softmax

    // 求当前线程负责的float4中, 四个float的最大值
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)  
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

    // warp shuffle, 归约求一行的最大值
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // 求当前线程负责的float4中, 四个float的和
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

    // warp shuffle, 归约求一行的和
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / row_sum;

    // 计算每个元素的softmax值
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // ===================== Top K ========================
    // 循环k次, 每次通过归约求和(warp shuffle), 计算出每行的softmax得分最大值top-1
    // 写入结果, 并将该位置置为负值, 下次循环可以求得top-2, 如此下去
    int start_col = first_elt_read_by_thread;  // (0,4,...,60)
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;  // 4*16=64

    // 循环k次
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
            // 求当前线程负责的4个元素中的softmax最大值, 以及对应的expert序号(0,4,...,60)+(0~3)
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // warp shuffle, 求一行中的最大值, 以及对应的expert序号
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // thread_group_idx范围是[0,15], 是一行中的线程序号
        // 这里使用第一个线程, 将结果写入output中
        if (thread_group_idx == 0)
        {
            const int idx = k * thread_row + k_idx;  // 第thread_row行, k_idx列
            output[idx] = max_val;                   // 当前行的top-k_idx
            indices[idx] = expert;                   // 当前行的top-k_idx对应的专家索引, 也就是col列号
            source_rows[idx] = k_idx * num_rows + thread_row;  // token_expert_indices索引表
        }

        // 如果不是最后一次循环, 则将
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;  // 0
            // 找到当前expert(列索引), 是哪个线程负责处理的
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;  // 0-15

            // 使用那个线程, 将对应位置值置为负数, 从而不影响下一次计算最大值
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // 设置为任意负值都可以, 因为row_chunk值是softmax后的, 介于 0 和 1 之间。
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }  // end for k_idx in [0, k)
}

/**
 * 计算每行的softmax值, 以及topk。
 * @tparam EXPERTS 专家数量
 * @tparam WARPS_PER_TB 每个线程块中使用的warp数量
 * @param input gating 的输出, shape=(seq_len, num_experts), 例如 (128, 64)
 * @param num_rows seq_len, 例如 128
 * @param k top-k, 例如 4
 * @param topk_weights 每个 token 对应的 topk 的 softmax 值
 * @param topk_indices 每个 token 对应的 topk 的 indice 值, 也就是专家序号
 * @param token_expert_indices 索引表, 形如
 * [[  0, 128, 256, 384],
 *  [  1, 129, 257, 385],
 *  [  2, 130, 258, 386],
 *			...
 *  [126, 254, 382, 510],
 *  [127, 255, 383, 511]])
*/
template <int EXPERTS, int WARPS_PER_TB>
void fused_softmax_topk(const float* input,
                        const int num_rows,
                        const int k,
                        float* topk_weights, 
                        int* topk_indices,
                        int* token_expert_indices)
{
    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, int(sizeof(float) * EXPERTS));  // 16
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);  // 16/4=4, 每次从Global Mem中向量化读取的float数, float4
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECs_PER_THREAD = std::max(1, int(EXPERTS / (ELTS_PER_LDG * WARP_SIZE))); // 64/(4*32)=0.5→1, 每个线程向量化读取的次数
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;  // 1*4, 每个线程处理的数据个数
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;  // 64/4=16, 一行使用16个线程处理
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;  // 32/16=2, 一个warp处理的行数

    const int num_warps  = CEIL(num_rows, ROWS_PER_WARP);  // 128/2=64, 一个warp处理2行，需要64个warp
    const int num_blocks = CEIL(num_warps, WARPS_PER_TB);  // 64/4=16, 每个block含4个warp，需要16个block
    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);  // block(32,4)

    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim>>>(
        input, num_rows, k, topk_weights, topk_indices, token_expert_indices);
}


torch::Tensor forward(torch::Tensor inputs, 
                      torch::Tensor gate_outputs,
                      const int k,
                      torch::Tensor m1, torch::Tensor m2,
                      torch::Tensor s1, torch::Tensor s2)

{
    torch::Device device(torch::kCUDA);

    // 1. fused_softmax_topk
    constexpr int EXPERTS = 64;
    constexpr int WARPS_PER_TB = 4;
    const int num_rows = gate_outputs.size(0);
    gate_outputs = gate_outputs.to(device);
    inputs = inputs.to(device);
    auto topk_weights = torch::zeros({num_rows, k}, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    auto topk_indices = torch::zeros({num_rows, k}, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    auto token_expert_indices = torch::zeros({num_rows, k}, torch::TensorOptions().dtype(torch::kInt32)).to(device);
    fused_softmax_topk<EXPERTS, WARPS_PER_TB>(gate_outputs.data_ptr<float>(), num_rows, k, 
                                              topk_weights.data_ptr<float>(),
                                              topk_indices.data_ptr<int>(),
                                              token_expert_indices.data_ptr<int>());

    // 2. expert_reset

    // 3. expert_copy

    // 4. compute_total_rows_before_expert

    // 5. do_expertv2

    // 6. moe_routing

    return token_expert_indices;
}