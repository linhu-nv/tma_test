/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholememory/device_reference.cuh>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "wholememory/integer_utils.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
//#include <cuda/pipeline>
#include <cuda/std/utility> // cuda::std::move


using barrier = cuda::barrier<cuda::thread_scope_block>;
//namespace cde = cuda::device::experimental;

inline __device__
void cp_async_bulk_global_to_shared(void *__dest, const void *__src, _CUDA_VSTD::uint32_t __size, ::cuda::barrier<::cuda::thread_scope_block> &__bar)
{
    //_LIBCUDACXX_DEBUG_ASSERT(__size % 16 == 0,   "Size must be multiple of 16.");
    //_LIBCUDACXX_DEBUG_ASSERT(__isShared(__dest), "Destination must be shared memory address.");
    //_LIBCUDACXX_DEBUG_ASSERT(__isGlobal(__src),  "Source must be global memory address.");

    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__dest))),
          "l"(static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__src))),
          "r"(__size),
          "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(::cuda::device::barrier_native_handle(__bar))))
        : "memory");
}

inline __device__
void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta; \n":::"memory");
}

inline __device__
void cp_async_bulk_shared_to_global(void *__dest, const void * __src, _CUDA_VSTD::uint32_t __size)
{
    //_LIBCUDACXX_DEBUG_ASSERT(__size % 16 == 0,   "Size must be multiple of 16.");
    //_LIBCUDACXX_DEBUG_ASSERT(__isGlobal(__dest), "Destination must be global memory address.");
    //_LIBCUDACXX_DEBUG_ASSERT(__isShared(__src),  "Source must be shared memory address.");

    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        :
        : "l"(static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__dest))),
          "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__src))),
          "r"(__size)
        : "memory");
}

inline __device__
void cp_async_bulk_commit_group()
{
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template <int n_prior>
inline __device__
void cp_async_bulk_wait_group_read()
{
  //static_assert(n_prior <= 63, "cp_async_bulk_wait_group_read: waiting for more than 63 groups is not supported.");
  asm volatile("cp.async.bulk.wait_group.read %0; \n"
               :
               : "n"(n_prior)
               : "memory");
}


//inline __device__
//_CUDA_VSTD::uint64_t * barrier_native_handle(barrier & b) {
//    return reinterpret_cast<_CUDA_VSTD::uint64_t *>(&b.__barrier);
//}

inline __device__
barrier::arrival_token barrier_arrive_tx(
    barrier & __b,
    //_CUDA_VSTD::ptrdiff_t __arrive_count_update,
    _CUDA_VSTD::ptrdiff_t __transaction_count_update) {

    //_LIBCUDACXX_DEBUG_ASSERT(__isShared(barrier_native_handle(__b)), "Barrier must be located in local shared memory.");
    //_LIBCUDACXX_DEBUG_ASSERT(1 <= __arrive_count_update, "Arrival count update must be at least one.");
    //_LIBCUDACXX_DEBUG_ASSERT(__arrive_count_update <= (1 << 20) - 1, "Arrival count update cannot exceed 2^20 - 1.");
    //_LIBCUDACXX_DEBUG_ASSERT(__transaction_count_update >= 0, "Transaction count update must be non-negative.");
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#contents-of-the-mbarrier-object
    //_LIBCUDACXX_DEBUG_ASSERT(__transaction_count_update <= (1 << 20) - 1, "Transaction count update cannot exceed 2^20 - 1.");

    barrier::arrival_token __token = {};
    // On architectures pre-sm90, arrive_tx is not supported.
    auto __bh = __cvta_generic_to_shared(cuda::device::barrier_native_handle(__b));
    asm (
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__token)
        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__bh)),
          "r"(static_cast<_CUDA_VSTD::uint32_t>(__transaction_count_update))
        : "memory");
    return __token;
}


namespace wholememory_ops {

template <typename DataTypeT>
__device__ __forceinline__ void mov_typed_data(DataTypeT* to, const DataTypeT* from)
{
  *to = *from;
}
template <int DATA_SIZE>
__device__ __forceinline__ void mov_data(void* to, const void* from)
{
  char* ptr_to         = static_cast<char*>(to);
  const char* ptr_from = static_cast<const char*>(from);
  for (int i = 0; i < DATA_SIZE; i++) {
    ptr_to[i] = ptr_from[i];
  }
}
template <typename DataTypeT, int DATA_SIZE>
struct typed_data_vector {
  DataTypeT data[DATA_SIZE];
};
template <>
struct typed_data_vector<double, 2> {
  double2 data;
};
template <>
struct typed_data_vector<int64_t, 2> {
  int4 data;
};
template <>
struct typed_data_vector<float, 2> {
  float2 data;
};
template <>
struct typed_data_vector<float, 4> {
  float4 data;
};
template <>
struct typed_data_vector<int, 2> {
  int2 data;
};
template <>
struct typed_data_vector<int, 4> {
  int4 data;
};
template <>
struct typed_data_vector<__half, 2> {
  __half2 data;
};
template <>
struct typed_data_vector<__half, 4> {
  int2 data;
};
template <>
struct typed_data_vector<__half, 8> {
  int4 data;
};
template <>
struct typed_data_vector<int16_t, 2> {
  int data;
};
template <>
struct typed_data_vector<int16_t, 4> {
  int2 data;
};
template <>
struct typed_data_vector<int16_t, 8> {
  int4 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 2> {
  nv_bfloat162 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 4> {
  int2 data;
};
template <>
struct typed_data_vector<nv_bfloat16, 8> {
  int4 data;
};
template <>
struct typed_data_vector<int8_t, 2> {
  int16_t data;
};
template <>
struct typed_data_vector<int8_t, 4> {
  int data;
};
template <>
struct typed_data_vector<int8_t, 8> {
  int2 data;
};
template <>
struct typed_data_vector<int8_t, 16> {
  int4 data;
};
template <typename DataTypeT, int DATA_SIZE>
__device__ __forceinline__ DataTypeT& typed_data_vector_at(
  typed_data_vector<DataTypeT, DATA_SIZE>& v, int idx)
{
  return ((DataTypeT*)(&v.data))[idx];
}

template <>
__device__ __forceinline__ void mov_data<1>(void* to, const void* from)
{
  mov_typed_data(static_cast<int8_t*>(to), static_cast<const int8_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<2>(void* to, const void* from)
{
  mov_typed_data(static_cast<int16_t*>(to), static_cast<const int16_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<4>(void* to, const void* from)
{
  mov_typed_data(static_cast<int32_t*>(to), static_cast<const int32_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<8>(void* to, const void* from)
{
  mov_typed_data(static_cast<int64_t*>(to), static_cast<const int64_t*>(from));
}
template <>
__device__ __forceinline__ void mov_data<16>(void* to, const void* from)
{
  mov_typed_data(static_cast<int4*>(to), static_cast<const int4*>(from));
}

template <typename DataTypeT>
class type_caster {
 public:
  using LoadTypeT  = DataTypeT;
  using StoreTypeT = DataTypeT;
  static __device__ __forceinline__ LoadTypeT convert_load_data(DataTypeT data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ DataTypeT convert_store_data(StoreTypeT data)
  {
    return static_cast<DataTypeT>(data);
  }
};
template <>
class type_caster<__half> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ __forceinline__ LoadTypeT convert_load_data(__half data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __forceinline__ __half convert_store_data(StoreTypeT data)
  {
    return static_cast<__half>(data);
  }
};
template <>
class type_caster<__nv_bfloat16> {
 public:
  using LoadTypeT  = float;
  using StoreTypeT = float;
  static __device__ LoadTypeT convert_load_data(__nv_bfloat16 data)
  {
    return static_cast<LoadTypeT>(data);
  }
  static __device__ __nv_bfloat16 convert_store_data(StoreTypeT data)
  {
    return static_cast<__nv_bfloat16>(data);
  }
};

template <typename FromT, typename ToT>
__device__ __forceinline__ ToT convert_type(FromT from)
{
  return type_caster<ToT>::convert_store_data(type_caster<FromT>::convert_load_data(from));
}

/**
 * Determine alignment of a WholeMemory matrix, in element count, maximum 16 / element_size.
 * @param embedding_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_wholememory_alignment_elt_count(
  wholememory_matrix_description_t embedding_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(embedding_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment = 16 / elt_size;
  for (; alignment > 1; alignment /= 2) {
    if (embedding_desc.storage_offset % alignment == 0 &&
        embedding_desc.sizes[1] % alignment == 0 && embedding_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

/**
 * Determine alignment of normal memory, in element count, maximum 16 / element_size.
 * @param ptr : pointer to the memory.
 * @param memory_desc : wholememory_matrix_description_t matrix description.
 * @return : Alignment that can be used, in element count.
 */
inline int determine_memory_alignment_elt_count(const void* ptr,
                                                wholememory_matrix_description_t memory_desc)
{
  int elt_size = static_cast<int>(wholememory_dtype_get_element_size(memory_desc.dtype));
  WHOLEMEMORY_CHECK(elt_size != -1);
  int alignment   = 16 / elt_size;
  int64_t int_ptr = reinterpret_cast<int64_t>(ptr);
  WHOLEMEMORY_CHECK(int_ptr % elt_size == 0);
  int_ptr /= elt_size;
  int_ptr += memory_desc.storage_offset;
  for (; alignment > 1; alignment /= 2) {
    if (int_ptr % alignment == 0 && memory_desc.sizes[1] % alignment == 0 &&
        memory_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

template <typename EmbeddingT, typename IndexT, typename OutputT, int ALIGNMENT = 1>
__global__ void gather_func_kernel(wholememory_gref_t embedding_gref,
                                   wholememory_matrix_description_t embedding_desc,
                                   const IndexT* indices,
                                   int64_t indice_count,
                                   OutputT* output,
                                   wholememory_matrix_description_t output_desc)
{
   __shared__ alignas(16) char shmem[6144];
  EmbeddingT* sh_buf = reinterpret_cast<EmbeddingT*>(shmem);
  __shared__ barrier bar;
  if (threadIdx.x == 0) { init(&bar, blockDim.x); }
  __syncwarp();
  
  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;

  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<OutputT, ALIGNMENT> outputs;
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);


  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  for (int64_t output_idx = warp_id; output_idx < indice_count; output_idx += gridDim.x * (blockDim.x / 32)) {
    OutputT* output_ptr = output + output_desc.storage_off + output_stride * output_idx;
    IndexT embedding_table_idx = indices[output_idx];
    if (embedding_table_idx < 0) continue;
    EmbeddingT* emb_ptr = 
        &embedding_dev_ref[embedding_desc.storage_offset + embedding_table_idx * embedding_stride];
    int copy_size = sizeof(EmbeddingT) * embedding_size;
    // Load data:
    uint64_t token;
    if (threadIdx.x == 0) {
        cp_async_bulk_global_to_shared(sh_buf, emb_ptr, copy_size, bar);
        token = barrier_arrive_tx(bar, copy_size);
    } else {
        token = bar.arrive();
    }
    bar.wait(cuda::std::move(token));
    //data convert in shared memory
    for (int emb_idx = threadIdx.x * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings, sh_buf + emb_idx);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(outputs, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(sh_buf + emb_idx, &outputs);
    }
    fence_proxy_async_shared_cta();
    __syncwarp();

    // Write back to global memory:
    if (threadIdx.x == 0) {
        cp_async_bulk_shared_to_global(output_ptr, sh_buf, copy_size);
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_group_read<0>();
    }
    __threadfence();
    __syncwarp();  
  }
  return ;
}

template <typename EmbeddingT, typename IndexT, typename OutputT>
void gather_temp_func(wholememory_gref_t embedding_gref,
                      wholememory_matrix_description_t embedding_desc,
                      void* indices,
                      int64_t indice_count,
                      void* output,
                      wholememory_matrix_description_t output_desc,
                      cudaStream_t stream,
                      int gather_sms)
{
  WHOLEMEMORY_EXPECTS(output_desc.sizes[0] == indice_count,
                      "gather_func, output shape[0]=%ld, but indice_count=%ld",
                      output_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(output, output_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);
  // int embedding_size = embedding_desc.sizes[1];
  // int thread_num       = wholememory::div_rounding_up_safe<int>(embedding_size, alignment);
  // thread_num           = std::min(thread_num, 512);
  // int64_t block_count = indice_count >= 1024 ? 1024 : static_cast<int>(indice_count);

  void (*kernel_fn)(wholememory_gref_t,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 16>;
      break;
    }
    case 8: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 8>;
      break;
    }
    case 4: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 4>;
      break;
    }
    case 2: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 2>;
      break;
    }
    case 1: {
      kernel_fn = gather_func_kernel<EmbeddingT, IndexT, OutputT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("gather func alignment=%d.", alignment);
      return;
    }
  }
  int block_size  = 32;
  int block_count = indice_count > 1568 ? 1568 : indice_count;
  if (gather_sms != -1) block_count = gather_sms;
  kernel_fn<<<block_count, block_size, 0, stream>>>(embedding_gref,
                                                    embedding_desc,
                                                    static_cast<const IndexT*>(indices),
                                                    indice_count,
                                                    static_cast<OutputT*>(output),
                                                    output_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

template <typename InputT, typename IndexT, typename EmbeddingT, int ALIGNMENT = 1>
__global__ void scatter_func_kernel(const InputT* input,
                                    wholememory_matrix_description_t input_desc,
                                    const IndexT* indices,
                                    int64_t indice_count,
                                    wholememory_gref_t embedding_gref,
                                    wholememory_matrix_description_t embedding_desc)
{
  auto block  = cooperative_groups::this_thread_block();
  auto mywarp = cooperative_groups::tiled_partition<32>(block);
  __shared__ char shm_in_char[24576];
  InputT* all_sh = reinterpret_cast<InputT*>(shm_in_char);
  InputT* my_shared;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  int lane_id = threadIdx.x % 32;

  int embedding_size       = embedding_desc.sizes[1];
  int64_t embedding_stride = embedding_desc.stride;
  int64_t input_stride     = input_desc.stride;
  int async_copy_align     = sizeof(InputT) > 4 ? 1 : 4 / sizeof(InputT);

  int shm_size = 24576 / sizeof(InputT);

  int batch_size = (shm_size / (blockDim.x / 32) - async_copy_align) /
                   input_stride;  // indices batch size in lines
  wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);

  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<InputT, ALIGNMENT> inputs;
  int input_off_tail =
    input_desc.storage_offset %
    async_copy_align;  // this is crutial for copy alignment, 4 bytes as alignment;
  bool use_shm = true;
  if (batch_size <= 0) {
    use_shm    = false;
    batch_size = 1;
  } else {
    my_shared = all_sh + shm_size / (blockDim.x / 32) * (threadIdx.x / 32);
  }
  for (int64_t input_idx = warp_id * batch_size; input_idx < indice_count;
       input_idx += gridDim.x * (blockDim.x / 32) * batch_size) {
    int cur_idx_lines =
      (indice_count - input_idx) > batch_size ? batch_size : indice_count - input_idx;
    const InputT* input_ptr =
      input + input_desc.storage_offset - input_off_tail + input_stride * input_idx;
    // this variable is also for alignment
    if (use_shm) {
      int copy_size = input_off_tail + cur_idx_lines * input_stride;
      if (input_idx + cur_idx_lines < indice_count)  // input_dim * sizeof(InputT) > 4 is needed
        copy_size = (copy_size + async_copy_align - 1) / async_copy_align * async_copy_align;
      copy_size *= sizeof(InputT);
      cooperative_groups::memcpy_async(mywarp, my_shared, input_ptr, copy_size);
      cooperative_groups::wait(mywarp);
    }
    for (int e = 0; e < cur_idx_lines; e++) {
      int64_t embedding_table_idx = indices[input_idx + e];
      if (embedding_table_idx < 0) continue;
      EmbeddingT* emb_ptr =
        &embedding_dev_ref[embedding_desc.storage_offset + embedding_table_idx * embedding_stride];

      for (int emb_idx = lane_id * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
        if (use_shm)
          mov_data<sizeof(InputT) * ALIGNMENT>(
            &inputs, my_shared + input_off_tail + e * input_stride + emb_idx);
        else
          mov_data<sizeof(InputT) * ALIGNMENT>(
            &inputs, input_ptr + input_off_tail + e * input_stride + emb_idx);
#pragma unroll
        for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
          typed_data_vector_at(embeddings, sub_idx) =
            convert_type<InputT, EmbeddingT>(typed_data_vector_at(inputs, sub_idx));
        }
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(emb_ptr + emb_idx, &embeddings);
      }
    }
    mywarp.sync();
  }
  return;
}

template <typename InputT, typename IndexT, typename EmbeddingT>
void scatter_temp_func(const void* input,
                       wholememory_matrix_description_t input_desc,
                       void* indices,
                       int64_t indice_count,
                       wholememory_gref_t embedding_gref,
                       wholememory_matrix_description_t embedding_desc,
                       cudaStream_t stream,
                       int scatter_sms)
{
  WHOLEMEMORY_EXPECTS(input_desc.sizes[0] == indice_count,
                      "scatter_func, input shape[0]=%ld, but indice_count=%ld",
                      input_desc.sizes[0],
                      indice_count);
  if (indice_count == 0 || embedding_desc.sizes[1] == 0) return;
  int wm_alignment = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment = determine_memory_alignment_elt_count(input, input_desc);
  int alignment    = std::min<int>(wm_alignment, mm_alignment);

  void (*kernel_fn)(const InputT*,
                    wholememory_matrix_description_t,
                    const IndexT*,
                    int64_t,
                    wholememory_gref_t,
                    wholememory_matrix_description_t) = nullptr;
  switch (alignment) {
    case 16: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 16>;
      break;
    }
    case 8: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 8>;
      break;
    }
    case 4: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 4>;
      break;
    }
    case 2: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 2>;
      break;
    }
    case 1: {
      kernel_fn = scatter_func_kernel<InputT, IndexT, EmbeddingT, 1>;
      break;
    }
    default: {
      WHOLEMEMORY_FAIL("scatter func alignment=%d.", alignment);
      return;
    }
  }
  int block_size  = 256;
  int block_count = indice_count > 1568 ? 1568 : indice_count;
  if (scatter_sms != -1) block_count = scatter_sms;
  kernel_fn<<<block_count, block_size, 0, stream>>>(static_cast<const InputT*>(input),
                                                    input_desc,
                                                    static_cast<const IndexT*>(indices),
                                                    indice_count,
                                                    embedding_gref,
                                                    embedding_desc);
  WM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace wholememory_ops
