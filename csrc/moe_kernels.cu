/*
 * Adapted from:
 * - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.8.0/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * - https://github.com/NVIDIA/TensorRT-LLM/blob/v0.8.0/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_template.h
 * Copyright (c) 2024, Tongkai Xu.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"

#define WARP_SIZE 32                        // 每个warp的中线程的数量
#define MAX_BYTES_PER_LDG 16                // 向量化访存最大字节数(16字节, float4)
#define CEIL(a, b) ((a + b - 1) / (b))


enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};


// ============================== moe_routing =================================

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, int RESIDUAL_NUM, bool HAS_BIAS, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(const T* expanded_permuted_rows, T* reduced_unpermuted_output, const T* skip_1,
    const T* skip_2, const T* bias, const float* scales, const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row, const int64_t cols, const int k, const int64_t num_valid_ptr)
{
    const int original_row = blockIdx.x;  // 一个block处理输出矩阵的一行
    const int num_rows = gridDim.x;       // block数量即行数
    const auto offset = original_row * cols;
    T* reduced_row_ptr = reduced_unpermuted_output + offset;  // 输出矩阵对应行的起始地址
    const T* skip_1_row_ptr{};
    const T* skip_2_row_ptr{};

    if (RESIDUAL_NUM >= 1)
    {
        skip_1_row_ptr = skip_1 + offset;
    }

    if (RESIDUAL_NUM == 2)
    {
        skip_2_row_ptr = skip_2 + offset;
    }
    const int64_t num_valid = num_valid_ptr;
    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)  // tid用于搬运每个元素
    {
        T thread_output{0.f};
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)  // for循环k次，加权求和
        {
            // 1. 当前行(original_row)的第 k_idx 个专家
            const int expanded_original_row = original_row + k_idx * num_rows;
            // 2. 查表dst_2_src_line, 找到其位于copy_outputs中的对应行
            const int expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            const int64_t k_offset = original_row * k + k_idx;
            const float row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row sum has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            // 3. copy_outputs中的对应行的起始地址
            const T* expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            const int expert_idx = expert_for_source_row[k_offset];

            const T* bias_ptr = bias + expert_idx * cols;
            const T bias_value = HAS_BIAS ? bias_ptr[tid] : T(0.f);

            // 4. 累加到thread_output中
            thread_output = static_cast<float>(thread_output)
                + row_scale * static_cast<float>(expanded_permuted_rows_row_ptr[tid] + bias_value);
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || thread_output))
        {
            assert(row_rescale != 0.f);
            thread_output = static_cast<float>(thread_output) / row_rescale;
        }

        if (RESIDUAL_NUM == 1)
        {
            thread_output = thread_output + skip_1_row_ptr[tid];
        }
        else if (RESIDUAL_NUM == 2)
        {
            thread_output = thread_output + skip_1_row_ptr[tid] + skip_2_row_ptr[tid];
        }
        reduced_row_ptr[tid] = thread_output;  // 累加值保存到输出
    }
}


template <typename T, int RESIDUAL_NUM>
void finalizeMoeRoutingKernelLauncherSelectBias(const T* expanded_permuted_rows, T* reduced_unpermuted_output,
    const T* skip_1, const T* skip_2, const T* bias, const float* scales,
    const int* expanded_source_row_to_expanded_dest_row, const int* expert_for_source_row, const int num_rows,
    const int64_t cols, const int k, const int64_t num_valid_ptr)
{
    const int blocks = num_rows;
    const int threads = std::min((int)cols, 1024);

    auto* const func = &finalizeMoeRoutingKernel<T, RESIDUAL_NUM, false, ScaleMode::DEFAULT, false>;
    func<<<blocks, threads>>>(expanded_permuted_rows, reduced_unpermuted_output, skip_1, skip_2, bias,
        scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

// ============================== gated activation =================================

template <class T, class ActFn>
__global__ void doGatedActivationKernel(
    T* output, const T* gemm_result, const int64_t num_valid_tokens, size_t inter_size)
{
    const int tid = threadIdx.x;
    const int token = blockIdx.x;
    if (token >= num_valid_tokens) return;

    ActFn fn{};
    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;
    for (int i = tid; i < inter_size; i += blockDim.x)
    {
        T gate_value = gemm_result[i];
        // BF16 isn't supported, use FP32 for activation function
        T gate_act = fn(gate_value);                    // 左半矩阵经过激活函数
        float fc1_value = gemm_result[i + inter_size];  // 右半矩阵保持不变
        output[i] = gate_act * fc1_value;               // 哈达玛积
    }
}

// ============================== do_expert =================================

struct EpilogueOpDefaultSilu {};
struct EpilogueOpDefault {};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op> 
struct Epilogue {};

constexpr auto DefaultScaleMode = cutlass::epilogue::thread::ScaleType::Default;

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultSilu>
{
    
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefault>
{
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerVectorAccess,
        ElementAccumulator, ElementAccumulator, DefaultScaleMode>;
};


template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void genericMoeGemmKernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    const int multi_processor_count)
{
    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;

    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t,
            WeightType>::type;

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    using EpilogueOp = typename Epilogue<ElementType,
        MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    // Finally, set up the kernel.
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
        typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
        typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,  // Stages=5
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kGroupScheduleMode>;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
    // TLLM_CHECK_WITH_INFO(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
    const int threadblock_count = multi_processor_count * occupancy;

    typename EpilogueOp::Params epilogue_op(
        ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

    const int group_size = gemm_k;
    typename GemmGrouped::Arguments args(num_experts, threadblock_count, group_size, epilogue_op,
        reinterpret_cast<const ElementType*>(A), reinterpret_cast<const CutlassWeightType*>(B),
        reinterpret_cast<const ElementType*>(weight_scales), reinterpret_cast<const ElementType*>(biases),
        reinterpret_cast<ElementType*>(C), total_rows_before_expert, gemm_n, gemm_k);

    GemmGrouped gemm;

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        printf("MoE FC kernel will fail for params.\n");
    }

    auto init_status = gemm.initialize(args);

    if (init_status != cutlass::Status::kSuccess) {
        printf("Failed to initialize cutlass variable batched gemm.\n");
    }
    auto run_status = gemm.run();
    if (run_status != cutlass::Status::kSuccess) {
        printf("Failed to run cutlass variable batched gemm.\n");
    }
}

// ============================== expert_copy =================================

__global__ void expandInputRowsKernel(const float* unpermuted_input, 
                                      float* permuted_output,
                                      const int* expanded_dest_row_to_expanded_source_row,
                                      int* expanded_source_row_to_expanded_dest_row,
                                      const int num_rows,
                                      const int64_t num_dest_rows, 
                                      const int cols)
{
    const int expanded_dest_row = blockIdx.x;  // 0-num_token * 4, 每个block搬运一行
    const int expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {   
        //dst_2_src_line
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = expanded_dest_row;
    }

    if (blockIdx.x < num_dest_rows)
    {
        const int source_row = expanded_source_row % num_rows;  // 当前线程要搬运inputs的哪一行

        const float* source_row_ptr = unpermuted_input + source_row * cols;  // inputs对应行的起始地址
        float* dest_row_ptr = permuted_output + expanded_dest_row * cols;  // 目标行的起始地址

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)  // 调动每个线程进行搬运，当线程数小于列数，一个线程将搬运多次
        {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

// ======================= compute_total_rows_before_expert ==========================

/**
 * @brief 二分查找。
*/
__device__ inline int findTotalEltsLeqTarget(const int* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}


__global__ void computeTotalRowsBeforeExpertKernel(const int* sorted_experts,
                                                   const int sorted_experts_len,
                                                   const int num_experts,
                                                   int64_t* total_rows_before_expert)
{
    // 每个专家用一个线程进行处理, 每个线程计算对应专家在 sorted_experts 中出现的总行数, 最终得到前缀表
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) return;

    // 二分查找
    total_rows_before_expert[expert] = findTotalEltsLeqTarget(sorted_experts, sorted_experts_len, expert);
}


// ============================== fused_softmax_topk =================================

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
    // 循环k次, 每次通过归约求max(warp shuffle), 计算出每行的softmax得分最大值top-1
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


// ============================== MoE Block =================================

torch::Tensor forward(torch::Tensor inputs, 
                      torch::Tensor gate_outputs,
                      const int k,
                      torch::Tensor fc1_expert_weights,
                      torch::Tensor fc2_expert_weights)

{
    torch::Device device(torch::kCUDA);

    inputs = inputs.to(device);
    gate_outputs = gate_outputs.to(device);
    fc1_expert_weights = fc1_expert_weights.to(device);
    fc2_expert_weights = fc2_expert_weights.to(device);

    // param
    constexpr int EXPERTS              = 64;
    constexpr int WARPS_PER_TB         = 4;
    const int num_rows                 = gate_outputs.size(0);
    const int64_t hidden_dim           = inputs.size(1);
    const int total_indices            = num_rows * k;
    const int64_t expanded_expert_rows = num_rows * k;
    const int64_t f1_out_rows          = num_rows * k;
    const int64_t fc1_out_size         = fc1_expert_weights.size(1);
    const int64_t fc2_out_size         = fc2_expert_weights.size(1);
    const int64_t inter_size           = fc1_out_size / 2;
    

    // allocate memory
    float *topk_weights;
    int *topk_indices, *token_expert_indices;
    int *sorted_topk_indices, *sorted_token_expert_indices;
    int64_t *total_rows_before_expert;
    float *copy_outputs;
    int* dst_2_src_line;
    float *glu_inter_result, *fc1_result, *fc2_result;
    cudaMalloc((void **)&topk_weights, num_rows * k * sizeof(float));                       // {num_rows, k}
    cudaMalloc((void **)&topk_indices, num_rows * k * sizeof(int));                         // {num_rows, k}
    cudaMalloc((void **)&token_expert_indices, num_rows * k * sizeof(int));                 // {num_rows, k}
    cudaMalloc((void **)&sorted_topk_indices, num_rows * k * sizeof(int));                  // {num_rows, k}
    cudaMalloc((void **)&sorted_token_expert_indices, num_rows * k * sizeof(int));          // {num_rows, k}
    cudaMalloc((void **)&total_rows_before_expert, EXPERTS * sizeof(int64_t));              // {EXPERTS}
    cudaMalloc((void **)&copy_outputs, num_rows * k * hidden_dim * sizeof(float));          // {num_rows * k, hidden_dim}, permuted_data
    cudaMalloc((void **)&dst_2_src_line, num_rows * k * sizeof(int));                       // {num_rows * k}
    cudaMalloc((void **)&glu_inter_result, num_rows * k * inter_size * 2 * sizeof(float));  // {num_rows * k, inter_size * 2}
    cudaMalloc((void **)&fc1_result, num_rows * k * inter_size * sizeof(float));            // {num_rows * k, inter_size}
    cudaMalloc((void **)&fc2_result, num_rows * k * fc2_out_size * sizeof(float));          // {num_rows * k, fc2_out_size}

    // 1. fused_softmax_topk
    fused_softmax_topk<EXPERTS, WARPS_PER_TB>(gate_outputs.data_ptr<float>(),
                                              num_rows,               // int
                                              k,                      // int, 
                                              topk_weights,           // {num_rows, k}, float
                                              topk_indices,           // {num_rows, k}, int
                                              token_expert_indices);  // {num_rows, k}, int

    // 2. expert_reset (RadixSort)
    void* sorter_ws = nullptr;
    size_t sorter_ws_size_bytes = 1;
    cudaMalloc(&sorter_ws, sorter_ws_size_bytes);
    cub::DeviceRadixSort::SortPairs(
        (void*) sorter_ws,            // workspace
        sorter_ws_size_bytes,         // workspace_size
        topk_indices,                 // keys_in,    {num_rows, k}, int
        sorted_topk_indices,          // keys_out,   {num_rows, k}, int
        token_expert_indices,         // values_in,  {num_rows, k}, int
        sorted_token_expert_indices,  // values_out, {num_rows, k}, int
        num_rows * k                  // num_key_value_pairs
    );

    // 3. compute_total_rows_before_expert
    const int threads = std::min(1024, EXPERTS);  // EXPERTS 个线程, 每个线程处理一个专家
    const int blocks  = CEIL(EXPERTS, threads);   // 对于64个专家, 1个block即可
    computeTotalRowsBeforeExpertKernel<<<blocks, threads>>>(sorted_topk_indices,        // {num_rows, k}, int
                                                            total_indices,              // num_rows * k
                                                            EXPERTS,
                                                            total_rows_before_expert);  // {EXPERTS}, int64_t

    // 4. expert_copy
    const int blocks2 = num_rows * k;
    const int threads2 = std::min((int)hidden_dim, 1024);
    expandInputRowsKernel<<<blocks2, threads2>>>(inputs.data_ptr<float>(),
                                                 copy_outputs,                 // {num_rows * k, hidden_dim}, float
                                                 sorted_token_expert_indices,  // {num_rows, k}, int
                                                 dst_2_src_line,               // {num_rows * k}
                                                 num_rows,
                                                 expanded_expert_rows,         // num_rows * k
                                                 hidden_dim);

    // 5. do_expert
    // 5.1 compute gemm_outputs1
    int multi_processor_count;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, 0);
    genericMoeGemmKernelLauncher<float,                                         // activation output type
                                 float,                                         // WeightType
                                 cutlass::arch::Sm80,
                                 EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<128, 128, 8>,         // ThreadblockShape
                                 cutlass::gemm::GemmShape<64, 64, 8>,           // WarpShape
                                 2>(                                            // Stages
                                copy_outputs,                                   // A, {num_rows * k, hidden_dim}, float
                                fc1_expert_weights.data_ptr<float>(),           // B
                                nullptr,                                        // weight_scales
                                nullptr,                                        // biases
                                glu_inter_result,                               // C, {num_rows * k, inter_size * 2}, float
                                total_rows_before_expert,                       // {EXPERTS}, int64_t
                                f1_out_rows,                                    // num_rows
                                fc1_out_size,                                   // gemm_n
                                hidden_dim,                                     // gemm_k
                                EXPERTS,
                                multi_processor_count);

    // 5.2 compute gated silu_outputs
    const int blocks3 = num_rows * k;
    const int threads3 = std::min((int)inter_size, 1024);
    auto* fn = &doGatedActivationKernel<float, cutlass::epilogue::thread::SiLu<float>>;
    fn<<<blocks3, threads3>>>(fc1_result,            // {num_rows * k, inter_size}
                              glu_inter_result,      // {num_rows * k, inter_size * 2}, float
                              expanded_expert_rows,  // num_rows * k
                              inter_size);
    
    // 5.3 compute gemm_outputs2
    genericMoeGemmKernelLauncher<float,                                         // activation output type
                                 float,                                         // WeightType
                                 cutlass::arch::Sm80,
                                 EpilogueOpDefault,
                                 cutlass::gemm::GemmShape<128, 128, 8>,         // ThreadblockShape
                                 cutlass::gemm::GemmShape<64, 64, 8>,           // WarpShape
                                 2>(                                            // Stages
                                fc1_result,                                     // A, {num_rows * k, inter_size}
                                fc2_expert_weights.data_ptr<float>(),           // B
                                nullptr,                                        // weight_scales
                                nullptr,                                        // biases
                                fc2_result,                                     // C, {num_rows * k, fc2_expert_weights.size(1)}
                                total_rows_before_expert,                       // {EXPERTS}, int64_t
                                expanded_expert_rows,                           // num_rows
                                hidden_dim,                                     // gemm_n, 是fc1_expert_weights的size(2)/2
                                inter_size,                                     // gemm_k
                                EXPERTS,               
                                multi_processor_count);

    // 6. moe_routing
    auto final_output = torch::zeros({num_rows, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat).device(device));
    finalizeMoeRoutingKernelLauncherSelectBias<float, 0>(
        fc2_result,
        final_output.data_ptr<float>(),
        nullptr,                         // skip_1
        nullptr,                         // skip_2
        nullptr,                         // fc2_expert_biases
        topk_weights,                    // expert_scales
        dst_2_src_line,                  // expanded_source_row_to_expanded_dest_row
        topk_indices,                    // expert_for_source_row
        num_rows,
        hidden_dim,                      // cols
        k,
        expanded_expert_rows);

    // free memory
    cudaFree(topk_weights);
    cudaFree(topk_indices); cudaFree(token_expert_indices);
    cudaFree(sorted_topk_indices); cudaFree(sorted_token_expert_indices);
    cudaFree(total_rows_before_expert);
    cudaFree(copy_outputs);
    cudaFree(dst_2_src_line);
    cudaFree(glu_inter_result); cudaFree(fc1_result); cudaFree(fc2_result);

    return final_output;
}

// ===================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}