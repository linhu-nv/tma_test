#include <stdlib.h>
#include <random>

#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/reverse.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>

using namespace std;

typedef uint32_t InputT;
typedef uint32_t OutputT;
typedef uint32_t EmbeddingT;
typedef int IndexT;

#define SCATTER
//#define GATHER

#ifdef SCATTER
//#define QUICK_VALIDATION_SCATTER
#define shm_size (24576/sizeof(InputT))//this setting assure best performance for 1024xint feature
#endif
#ifdef GATHER
//#define QUICK_VALIDATION_GATHER
#define shm_size (16384/sizeof(InputT))//TODO this may be important, can be fine-tuned
                                       //in our experiments, less shm size may be better
#endif

//#define NAIVE_COPY
#define TMA_COPY
//#define TMA_PIPELINE_COPY

//#define shm_size (49152/sizeof(InputT))

struct desc{
  int size;
  int dim;
  int stride;
  int start_off;
  desc(int _s, int _d, int _stride, int _startoff):
        size(_s), dim(_d), stride(_stride), start_off(_startoff){}
};

#define CUDA_TRY(call)                                                          \
  do {                                                                          \
    cudaError_t const status = (call);                                          \
    if (cudaSuccess != status) {                                                \
      printf("%s %s %d\n", cudaGetErrorString(status), __FILE__, __LINE__);  \
    }                                                                           \
  } while (0)

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
};//WARNING: A BUG MAYBE
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
inline int determine_wholememory_alignment_elt_count(desc embedding_desc)
{
  int elt_size = static_cast<int>(sizeof(EmbeddingT));
  assert(elt_size != -1);
  int alignment = 16 / elt_size;
  for (; alignment > 1; alignment /= 2) {
    if (embedding_desc.start_off % alignment == 0 &&
        embedding_desc.dim % alignment == 0 && embedding_desc.stride % alignment == 0)
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
inline int determine_memory_alignment_elt_count(const void* ptr, desc memory_desc)
{
  int elt_size = static_cast<int>(sizeof(InputT));
  assert(elt_size != -1);
  int alignment   = 16 / elt_size;
  int64_t int_ptr = reinterpret_cast<int64_t>(ptr);
  assert(int_ptr % elt_size == 0);
  int_ptr /= elt_size;
  int_ptr += memory_desc.start_off;
  for (; alignment > 1; alignment /= 2) {
    if (int_ptr % alignment == 0 && memory_desc.dim % alignment == 0 &&
        memory_desc.stride % alignment == 0)
      break;
  }
  return alignment;
}

template <int ALIGNMENT = 3>
__global__ void scatter_func_kernel(const InputT* input,
                                    desc input_desc,
                                    const IndexT* indices,
                                    int indice_count,
                                    EmbeddingT* embedding,
                                    desc embedding_desc)
{
  int64_t input_idx          = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  int thread_idx             = threadIdx.x;
  IndexT embedding_table_idx = indices[input_idx];
  if (embedding_table_idx < 0) return;
  //wholememory::device_reference<EmbeddingT> embedding_dev_ref(embedding_gref);
  int embedding_size       = embedding_desc.dim;
  int64_t embedding_stride = embedding_desc.stride;
  int64_t input_stride     = input_desc.stride;
  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<InputT, ALIGNMENT> inputs;
  const InputT* input_ptr  = input + input_desc.start_off + input_stride * input_idx;
  int64_t embedding_offset = embedding_desc.start_off + embedding_table_idx * embedding_stride;
  for (; input_idx < indice_count; input_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(InputT) * ALIGNMENT>(&inputs, input_ptr + emb_idx);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(embeddings, sub_idx) =
          convert_type<InputT, EmbeddingT>(typed_data_vector_at(inputs, sub_idx));
      }
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(embedding + embedding_offset + emb_idx,
                                               &embeddings);
    }
  }
}

__inline__ __device__
void cp_async_bulk_global_to_shared(void *__dest, const void *__src, _CUDA_VSTD::uint32_t __size, ::cuda::barrier<::cuda::thread_scope_block> &__bar)
{
    assert(__size % 16 == 0);
    assert(__isShared(__dest));
    assert(__isGlobal(__src));

    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__dest))),
          "l"(static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__src))),
          "r"(__size),
          "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(::cuda::device::barrier_native_handle(__bar))))
        : "memory");
}

template <int ALIGNMENT = 3>
__global__ void scatter_kernel_TMA(const InputT* input,
                                    desc input_desc,
                                    const IndexT* indices,
                                    int indice_count,
                                    EmbeddingT* embedding,
                                    desc embedding_desc)
{
  
  auto block = cooperative_groups::this_thread_block();
  auto mywarp = cooperative_groups::tiled_partition<32>(block);
  extern __shared__ InputT shared[];
  InputT* my_shared;
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x)/32;
  int lane_id = threadIdx.x % 32;

  using barrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  block.sync();

  int embedding_size       = embedding_desc.dim;
  int64_t embedding_stride = embedding_desc.stride;
  int block_idx = block.group_index().x;
  int64_t input_stride     = input_desc.stride;
  int async_copy_align = 16;
  int batch_size = (shm_size/(blockDim.x/32)-async_copy_align)/input_stride;//indices batch size in lines
  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<InputT, ALIGNMENT> inputs;
  int input_off_tail = input_desc.start_off%async_copy_align;//this is crutial for copy alignment, 4 bytes as alignment;
  bool use_shm = true;
  if (batch_size <= 0) { 
    use_shm = false; batch_size = 1;
  }else {
    my_shared = shared + shm_size/(blockDim.x/32)*(threadIdx.x/32);
  }
  for (int64_t input_idx = warp_id*batch_size; input_idx < indice_count; input_idx += gridDim.x*(blockDim.x/32)*batch_size) {
	  int cur_idx_lines = (indice_count - input_idx) > batch_size ? batch_size : indice_count - input_idx;
	  const InputT* input_ptr = input + input_desc.start_off - input_off_tail + input_stride * input_idx;
    //this variable is also for alignment
    if (use_shm) {
      int copy_size = (input_off_tail + cur_idx_lines*input_stride)*sizeof(InputT);
      if (input_idx + cur_idx_lines < indice_count)//input_dim * sizeof(InputT) > 4 is needed
        copy_size = (copy_size + async_copy_align -1)/async_copy_align*async_copy_align;
      cp_async_bulk_global_to_shared((void *)my_shared, (void *)input_ptr, copy_size, bar);
	    //cooperative_groups::memcpy_async(mywarp, my_shared, input_ptr, copy_size);
	    //cooperative_groups::wait(mywarp);

    }
	  for (int e = 0; e < cur_idx_lines; e ++) {
		  int64_t embedding_table_idx = indices[input_idx + e];
	  	EmbeddingT *emb_ptr = embedding + embedding_desc.start_off + embedding_table_idx*embedding_stride;
      
      for (int emb_idx = lane_id * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
        if (use_shm) mov_data<sizeof(InputT) * ALIGNMENT>(&inputs, my_shared + input_off_tail + e*input_stride + emb_idx);
        else mov_data<sizeof(InputT) * ALIGNMENT>(&inputs, input_ptr + input_off_tail + e*input_stride + emb_idx);
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
  return ;
}


template<int ALIGNMENT = 3,uint8_t stage_count = 2>//TODO set stage count to 2
__global__ void scatter_kernel_TMA_pipeline(const InputT* input,
                                    desc input_desc,
                                    const IndexT* indices,
                                    int indice_count,
                                    EmbeddingT* embedding,
                                    desc emb_desc)
{
  //auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  extern __shared__ EmbeddingT shared[]; // stages_count * block.size() * sizeof(int) bytes
  
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,stage_count> shared_state;
  auto pipeline = cuda::make_pipeline(block, &shared_state);

  int input_stride = input_desc.stride;
  int batch_size = shm_size/stage_count/input_stride;
  int block_idx = block.group_index().x;
  int my_batch_num = ((indice_count+batch_size-1)/batch_size) % gridDim.x > block_idx ? 1 : 0;
  my_batch_num += indice_count/(batch_size*gridDim.x);
  
  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<InputT, ALIGNMENT> inputs;

  for (int get_batch = 0, put_batch = 0; put_batch < my_batch_num; put_batch ++) {
    for (; get_batch < my_batch_num && get_batch < (put_batch + stage_count); get_batch ++) {
      int64_t input_idx = (get_batch * gridDim.x + block_idx) * batch_size;
      int idx_line_num = (indice_count - input_idx) > batch_size ? batch_size : (indice_count - input_idx);
      int shared_off = (get_batch % stage_count)*batch_size*input_stride;
      EmbeddingT *in_addr = (EmbeddingT*)(input + input_desc.start_off + input_stride * input_idx);
      pipeline.producer_acquire();
      cuda::memcpy_async(block, shared+shared_off, in_addr, sizeof(EmbeddingT)*idx_line_num*input_stride, pipeline);
      pipeline.producer_commit();
    }
    int64_t input_idx = (put_batch * gridDim.x + block_idx) * batch_size;
    int idx_line_num = (indice_count - input_idx) > batch_size ? batch_size : (indice_count - input_idx);
    pipeline.consumer_wait();
    for (int e = 0; e < idx_line_num; e ++) {
      int64_t emb_idx = indices[input_idx+e];
      EmbeddingT* out_addr = embedding + emb_desc.start_off + emb_desc.stride*emb_idx;
      int shared_off = (put_batch % stage_count)* batch_size*input_stride;
      /*for (int emb_idx = block.thread_rank(); emb_idx < emb_desc.dim; emb_idx += block.size()) {
        out_addr[emb_idx] = shared[shared_off+e*input_stride+emb_idx];
      }*/
      for (int emb_idx = block.thread_rank() * ALIGNMENT; emb_idx < emb_desc.dim; emb_idx += ALIGNMENT * block.size()) {
        mov_data<sizeof(InputT) * ALIGNMENT>(&inputs, shared + shared_off + e*input_stride + emb_idx);
#pragma unroll
        for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
          typed_data_vector_at(embeddings, sub_idx) =
            convert_type<InputT, EmbeddingT>(typed_data_vector_at(inputs, sub_idx));
        }
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(out_addr + emb_idx, &embeddings);
      }
    }
    //block.sync();
    pipeline.consumer_release();
    block.sync();
  }
  return ;
}

void scatter_temp_func(InputT* input,
                       struct desc input_desc,
                       IndexT* indices,
                       int indice_count,
                       EmbeddingT* embedding,
                       struct desc embedding_desc)
{
  if (indice_count == 0 || embedding_desc.dim == 0) return;
  int wm_alignment   = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment   = determine_memory_alignment_elt_count(input, input_desc);
  int alignment      = std::min<int>(wm_alignment, mm_alignment);
  int embedding_size = embedding_desc.dim;
#ifdef NAIVE_COPY
  int thread_x       = (embedding_size + alignment-1)/alignment;
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  int64_t block_count_64 = (indice_count + thread_y - 1) / thread_y;
  int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);
  void (*kernel_fn)(const InputT*,
                    desc,
                    const IndexT*,
                    int,
                    EmbeddingT*,
                    desc) = nullptr;
  //printf("key parameters: %d %d %d %d\n",thread_x, thread_y, block_count, alignment);
  switch (alignment) {
    case 16: {kernel_fn = scatter_func_kernel<16>;break;}
    case 8: {kernel_fn = scatter_func_kernel<8>;break;}
    case 4: {kernel_fn = scatter_func_kernel<4>;break;}
    case 2: {kernel_fn = scatter_func_kernel<2>;break;}
    case 1: {kernel_fn = scatter_func_kernel<1>;break;}
    default: {
      printf("scatter func alignment=%d.\n", alignment);
      return;
    }
  }
#endif
  cudaEvent_t start, stop;
	float esp_time_gpu;
	CUDA_TRY(cudaEventCreate(&start));
	CUDA_TRY(cudaEventCreate(&stop));
  CUDA_TRY(cudaEventRecord(start, 0));
#ifdef NAIVE_COPY
  kernel_fn<<<block_count, block_dim>>>(input,
                                        input_desc,
                                        indices,
                                        indice_count,
                                        embedding,
                                        embedding_desc);
#endif
#ifdef TMA_COPY
  void (*kernel_fn)(const InputT*,
                    desc,
                    const IndexT*,
                    int,
                    EmbeddingT*,
                    desc) = nullptr;
  //printf("key parameters: %d %d %d %d\n",thread_x, thread_y, block_count, alignment);
  switch (alignment) {
    case 16: {  kernel_fn = scatter_kernel_TMA<16>; break;}
    case 8: {  kernel_fn = scatter_kernel_TMA<8>; break;}
    case 4: {  kernel_fn = scatter_kernel_TMA<4>; break;}
    case 2: {  kernel_fn = scatter_kernel_TMA<2>; break;}
    case 1: {  kernel_fn = scatter_kernel_TMA<1>; break;}
    default: {
      printf("scatter func alignment=%d.\n", alignment); return;
    }
  }
  int block_size = (embedding_desc.dim + alignment-1)/alignment;
  //block_size = block_size > 512 ? 512 : block_size;
  block_size = 256;
  int block_count = indice_count > 2048 ? 2048 : indice_count;
  kernel_fn<<<block_count, block_size, shm_size*sizeof(InputT)>>>(input,
                                                                          input_desc,
                                                                          indices,
                                                                          indice_count,
                                                                          embedding,
                                                                          embedding_desc);
#endif
#ifdef TMA_PIPELINE_COPY
   void (*kernel_fn)(const InputT*,
                    desc,
                    const IndexT*,
                    int,
                    EmbeddingT*,
                    desc) = nullptr;
  //printf("key parameters: %d %d %d %d\n",thread_x, thread_y, block_count, alignment);
  switch (alignment) {
    case 16: {  kernel_fn = scatter_kernel_TMA_pipeline<16,2>; break;}
    case 8: {  kernel_fn = scatter_kernel_TMA_pipeline<8,2>; break;}
    case 4: {  kernel_fn = scatter_kernel_TMA_pipeline<4,2>; break;}
    case 2: {  kernel_fn = scatter_kernel_TMA_pipeline<2,2>; break;}
    case 1: {  kernel_fn = scatter_kernel_TMA_pipeline<1,2>; break;}
    default: {
      printf("scatter func alignment=%d.\n", alignment); return;
    }
  }
  int block_size = (embedding_desc.dim + alignment-1)/alignment;
  block_size = block_size > 256 ? 256 : block_size;
  int block_count = indice_count > 4096 ? 4096 : indice_count;
  kernel_fn<<<block_count, block_size, shm_size*sizeof(EmbeddingT)>>>(input,
                                                                   input_desc,
                                                                   indices,
                                                                   indice_count,
                                                                   embedding,
                                                                   embedding_desc);

#endif
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaEventRecord(stop, 0));
	CUDA_TRY(cudaEventSynchronize(stop));
  CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
	printf("    Time for the kernel is: %f ms, where alignment is %d\n", esp_time_gpu, alignment);
  return ;
}

template <int ALIGNMENT = 1>
__global__ void gather_func_kernel(EmbeddingT* embedding,
                                   desc embedding_desc,
                                   const IndexT* indices,
                                   int64_t indice_count,
                                   OutputT* output,
                                   desc output_desc)
{
  int64_t output_idx         = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  IndexT embedding_table_idx = indices[output_idx];
  if (embedding_table_idx < 0) return;
  int thread_idx           = threadIdx.x;
  int embedding_size       = embedding_desc.dim;
  int64_t embedding_stride = embedding_desc.stride;
  int64_t output_stride    = output_desc.stride;
  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<OutputT, ALIGNMENT> outputs;
  OutputT* output_ptr      = output + output_desc.start_off + output_stride * output_idx;
  int64_t embedding_offset = embedding_desc.start_off + embedding_table_idx * embedding_stride;
  for (; output_idx < indice_count; output_idx += static_cast<int64_t>(gridDim.x) * blockDim.y) {
    for (int emb_idx = thread_idx * ALIGNMENT; emb_idx < embedding_size;
         emb_idx += ALIGNMENT * blockDim.x) {
      mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings, embedding + embedding_offset + emb_idx);
#pragma unroll
      for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
        typed_data_vector_at(outputs, sub_idx) =
          convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
      }
      mov_data<sizeof(OutputT) * ALIGNMENT>(output_ptr + emb_idx, &outputs);
    }
  }
}

template <int ALIGNMENT = 1>
__global__ void gather_func_kernel_TMA(EmbeddingT* embedding,
                                   desc embedding_desc,
                                   const IndexT* indices,
                                   int64_t indice_count,
                                   OutputT* output,
                                   desc output_desc)
{
  auto block = cooperative_groups::this_thread_block();
  auto mywarp = cooperative_groups::tiled_partition<32>(block);
  extern __shared__ OutputT shared[];
  OutputT* my_shared; 
  int warp_id = (threadIdx.x + blockIdx.x * blockDim.x)/32;
  int lane_id = threadIdx.x % 32;

  int embedding_size       = embedding_desc.dim;
  int64_t embedding_stride = embedding_desc.stride;
  int block_idx = block.group_index().x;
  int64_t output_stride     = output_desc.stride;
  //int async_copy_align = 4;
  int batch_size = shm_size/(blockDim.x/32)/output_stride;//indices batch size for a block in lines
  
  typed_data_vector<EmbeddingT, ALIGNMENT> embeddings;
  typed_data_vector<OutputT, ALIGNMENT> outputs;

  bool use_shm = true;
  if (batch_size <= 0) {
    use_shm = false;
    batch_size = 1;
  } else {
    my_shared = shared + shm_size/(blockDim.x/32)*(threadIdx.x/32);
  }
  //int output_off_tail = output_desc.storage_offset%async_copy_align;//this is crutial for copy alignment, 4 bytes as alignment;
  
  for (int64_t output_idx = warp_id*batch_size; output_idx < indice_count; output_idx += gridDim.x*(blockDim.x/32)*batch_size) {
	  int cur_idx_lines = (indice_count - output_idx) > batch_size ? batch_size : indice_count - output_idx;
    OutputT* output_ptr = output + output_desc.start_off + output_stride * output_idx;
    if (!use_shm) {
      my_shared = output_ptr;
    }
    for (int e = 0; e < cur_idx_lines; e ++) {
		  int64_t embedding_table_idx = indices[output_idx + e];
	  	EmbeddingT *emb_ptr = embedding + embedding_desc.start_off + embedding_table_idx*embedding_stride;
      
      for (int emb_idx = lane_id * ALIGNMENT; emb_idx < embedding_size; emb_idx += ALIGNMENT * 32) {
        mov_data<sizeof(EmbeddingT) * ALIGNMENT>(&embeddings, emb_ptr + emb_idx);
#pragma unroll
        for (int sub_idx = 0; sub_idx < ALIGNMENT; sub_idx++) {
          typed_data_vector_at(outputs, sub_idx) =
            convert_type<EmbeddingT, OutputT>(typed_data_vector_at(embeddings, sub_idx));
        }
        mov_data<sizeof(InputT) * ALIGNMENT>(my_shared + e*output_stride + emb_idx, &outputs);
      }
	  }
    //block.sync();
    if (use_shm) {
      //this variable is also for alignment
      int copy_size = cur_idx_lines*output_stride*sizeof(OutputT);
      //if (input_idx + cur_idx_lines < indice_count)//input_dim * sizeof(InputT) > 4 is needed
      //  copy_size = (copy_size + async_copy_align - 1)/async_copy_align*async_copy_align;
	    cooperative_groups::memcpy_async(mywarp, output_ptr, my_shared, copy_size);
	    cooperative_groups::wait(mywarp);
    } 
  }
  return;
}


void gather_temp_func(EmbeddingT *embedding,
                      desc embedding_desc,
                      IndexT* indices,
                      int64_t indice_count,
                      OutputT* output,
                      desc output_desc)
{
  int wm_alignment   = determine_wholememory_alignment_elt_count(embedding_desc);
  int mm_alignment   = determine_memory_alignment_elt_count(output, output_desc);
  int alignment      = std::min<int>(wm_alignment, mm_alignment);
  int embedding_size = embedding_desc.dim;
#ifdef NAIVE_COPY
  int thread_x       = (embedding_size+alignment-1)/alignment*alignment;
  thread_x           = std::min(thread_x, 256);
  int thread_y       = 1;
  if (thread_x < 64) {
    int power2_thread_x = 1;
    for (; power2_thread_x < thread_x; power2_thread_x *= 2)
      ;
    thread_x = power2_thread_x;
    thread_y = 64 / thread_x;
  }
  int64_t block_count_64 = (indice_count + thread_y - 1) / thread_y;
  int block_count = block_count_64 >= INT_MAX ? INT_MAX / 4 : static_cast<int>(block_count_64);
  dim3 block_dim(thread_x, thread_y, 1);
  void (*kernel_fn)(EmbeddingT*,
                    desc,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    desc) = nullptr;
  switch (alignment) {
    case 16: { kernel_fn = gather_func_kernel<16>; break;}
    case 8: { kernel_fn = gather_func_kernel<8>; break;}
    case 4: { kernel_fn = gather_func_kernel<4>; break;}
    case 2: { kernel_fn = gather_func_kernel<2>; break;}
    case 1: { kernel_fn = gather_func_kernel<1>; break;}
    default: {
      printf("gather func alignment=%d.", alignment);
      return;
    }
  }
#endif
#ifdef TMA_COPY
  int thread_num = (embedding_size+alignment-1)/ alignment;
  //thread_num = std::min(thread_num, 512);
  thread_num = 256;
  int64_t block_count = indice_count >= 2048 ? 2048 : static_cast<int>(indice_count);
  
  void (*kernel_fn)(EmbeddingT*,
                    desc,
                    const IndexT*,
                    int64_t,
                    OutputT*,
                    desc) = nullptr;
  switch (alignment) {
    case 16: { kernel_fn = gather_func_kernel_TMA<16>; break;}
    case 8: { kernel_fn = gather_func_kernel_TMA<8>; break;}
    case 4: { kernel_fn = gather_func_kernel_TMA<4>; break;}
    case 2: { kernel_fn = gather_func_kernel_TMA<2>; break;}
    case 1: { kernel_fn = gather_func_kernel_TMA<1>; break;}
    default: {
      printf("gather func alignment=%d.", alignment);
      return;
    }
  }
  
#endif
  cudaEvent_t start, stop;
	float esp_time_gpu;
	CUDA_TRY(cudaEventCreate(&start));
	CUDA_TRY(cudaEventCreate(&stop));
  CUDA_TRY(cudaEventRecord(start, 0));
#ifdef NAIVE_COPY
  kernel_fn<<<block_count, block_dim>>>(embedding,
                                        embedding_desc,
                                        indices,
                                        indice_count,
                                        output,
                                        output_desc);
#endif
#ifdef TMA_COPY
  kernel_fn<<<block_count, thread_num, shm_size*sizeof(OutputT)>>>(embedding,
                                                                  embedding_desc,
                                                                  indices,
                                                                  indice_count,
                                                                  output,
                                                                  output_desc);
#endif
  CUDA_TRY(cudaDeviceSynchronize());
  CUDA_TRY(cudaEventRecord(stop, 0));
	CUDA_TRY(cudaEventSynchronize(stop));
  CUDA_TRY(cudaEventElapsedTime(&esp_time_gpu, start, stop));
	printf("    Time for the kernel is: %f ms, where alignment is %d\n", esp_time_gpu, alignment);
  CUDA_TRY(cudaGetLastError());
}


int main (int argc, char**argv) {
  //key parameters
  int embedding_dim = 128;
  if (argc > 1) embedding_dim = atoi(argv[1]);
  int emb_start_off = 0;//the offset is also in element
  if (argc > 2) emb_start_off = atoi(argv[2]);
  int input_start_off = 0;//emb_start_off;
  int output_start_off = 0;//
  if (argc > 3) {
    input_start_off = atoi(argv[3]);
    output_start_off = atoi(argv[3]);
  }
  int output_dim = embedding_dim;
  int input_dim = embedding_dim;
  uint64_t embedding_size = 10 * 1024UL * 1024UL;
  uint64_t input_size = embedding_size/2;
  uint64_t output_size = embedding_size/2;
  printf("the embedding dim is %d, emb_start_off %d, input/output_start_off %d\n", embedding_dim, emb_start_off, input_start_off);

  uint64_t total_size_gb = (embedding_size + input_size)*embedding_dim*sizeof(EmbeddingT)/1024/1024/1024;
  printf("the total size is %d GB\n", total_size_gb);
#ifdef SCATTER
  //construct input
  InputT *input;
  int in_aligned_size = 16/sizeof(InputT);
  int in_stride = input_dim % in_aligned_size == 0 ? 
                     input_dim : (input_dim/in_aligned_size+1)*in_aligned_size;
  int64_t in_malloc_size = (int64_t)in_stride * input_size + input_start_off;
  CUDA_TRY(cudaMalloc((void **)&input, sizeof(InputT)*in_malloc_size));
  //printf("the input stride is %d, the input_malloc_size is %ld, ptr is 0x%p\n", in_stride, in_malloc_size, input);

  thrust::sequence(thrust::device, input+input_start_off, input+in_malloc_size, 0);//NOTE: more initialization methods needed
  thrust::reverse(thrust::device, input+input_start_off, input+in_malloc_size);
  struct desc input_desc = desc(input_size, input_dim, in_stride, input_start_off);
  printf("construct input tensor done, the in_stride is %d\n", in_stride);
#endif

  //construct embedding
  EmbeddingT * embedding;
  int emb_aligned_size = 16/sizeof(EmbeddingT);
  int emb_stride = embedding_dim % emb_aligned_size == 0 ? 
                     embedding_dim : (embedding_dim/emb_aligned_size+1)*emb_aligned_size;
  int64_t emb_malloc_size = (int64_t)emb_stride * embedding_size + emb_start_off;
  CUDA_TRY(cudaMalloc((void **)&embedding, sizeof(EmbeddingT)*emb_malloc_size));
  //printf("the emb stride is %d, the emb_malloc_size is %ld, the ptr is 0x%p\n", emb_stride, emb_malloc_size, embedding);

  thrust::sequence(thrust::device, embedding+emb_start_off, embedding+emb_malloc_size, 0);
  struct desc emb_desc = desc(embedding_size, embedding_dim, emb_stride, emb_start_off);
  printf("construct the target embedding done, the emb_stride is %d\n", emb_stride);

#ifdef GATHER
  //construct output;
  OutputT* output;
  int out_aligned_size = 16/sizeof(OutputT);
  int out_stride = output_dim % out_aligned_size == 0 ? 
                     output_dim : (output_dim/out_aligned_size+1)*out_aligned_size;
  int64_t out_malloc_size = (int64_t)out_stride * output_size + output_start_off;
  CUDA_TRY(cudaMalloc((void **)&output, sizeof(OutputT)*out_malloc_size));
  //printf("the input stride is %d, the input_malloc_size is %ld, ptr is 0x%p\n", in_stride, in_malloc_size, input);
  struct desc output_desc = desc(output_size, output_dim, out_stride, output_start_off);
  printf("construct output tensor done, the out_stride is %d\n", out_stride);
#endif

  //construct indices
  IndexT *indices;
  CUDA_TRY(cudaMalloc((void **)&indices, sizeof(IndexT)*input_size));
  IndexT *h_indices = (IndexT*)malloc(sizeof(IndexT)*input_size);
#ifdef QUICK_VALIDATION_SCATTER 
  EmbeddingT* h_embedding = (EmbeddingT*)malloc(sizeof(EmbeddingT)*emb_malloc_size);
  InputT* h_input = (InputT *)malloc(sizeof(InputT)*in_malloc_size);
#endif
#ifdef QUICK_VALIDATION_GATHER
  EmbeddingT* h_embedding = (EmbeddingT*)malloc(sizeof(EmbeddingT)*emb_malloc_size);
  OutputT* h_output = (OutputT *)malloc(sizeof(OutputT)*out_malloc_size);
#endif
  uint8_t* used = (uint8_t*)malloc(sizeof(uint8_t)*embedding_size); 
  for (int iter = 0; iter < 2; iter ++) {
    printf("    start generating the indices for %d th iteration...\n", iter);
    //different iterations have different indices
    int min = 0, max = embedding_size-1;
    random_device seed;
	  ranlux48 engine(seed());
    uniform_int_distribution<> distrib(min, max);
    memset(used, 0, sizeof(uint8_t)*embedding_size);
    for (int i = 0; i < input_size; i ++) {
      int random = distrib(engine);//随机数
      while(used[random]) {
        random = distrib(engine);
      }
      used[random] = 1;
      //NOTE: currently only int/half_int is supported
      h_indices[i] = (IndexT)random;
    }
    thrust::sort(thrust::host, h_indices, h_indices + input_size);
    CUDA_TRY(cudaMemcpy(indices, h_indices, sizeof(IndexT)*input_size, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaDeviceSynchronize());
    printf("    indices prepared, start the scatter function now...\n");
#ifdef SCATTER
    scatter_temp_func(input,
                      input_desc,
                      indices,
                      input_size,
                      embedding,
                      emb_desc);
#endif
#ifdef GATHER
    gather_temp_func(embedding,
                     emb_desc,
                     indices,
                     input_size,
                     output,
                     output_desc);
#endif
#ifdef QUICK_VALIDATION_GATHER
    CUDA_TRY(cudaMemcpy(h_embedding, embedding, sizeof(EmbeddingT)*emb_malloc_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_indices, indices, sizeof(IndexT)*output_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_output, output, sizeof(OutputT)*out_malloc_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaDeviceSynchronize());
    bool passed = true;
    for (int64_t i = 0; i < input_size; i ++) {
      int64_t emb_idx = h_indices[i];
      if (h_output[i*out_stride+output_start_off] != emb_idx*emb_stride) {
        passed = false;
        printf("i = %lu, the first ele of output is %d, should be %d\n",h_output[i*out_stride+output_start_off], emb_idx*emb_stride);
        break;
      }
    }
    if (passed)
      printf("    the %d th iteration passed quick validation!\n", iter);
    else 
      printf("    the %d th iteration did NOT pass the quick validation!\n", iter);
#endif
#ifdef QUICK_VALIDATION_SCATTER//NOTE the check here is designed for int type
    CUDA_TRY(cudaMemcpy(h_embedding, embedding, sizeof(EmbeddingT)*emb_malloc_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_indices, indices, sizeof(IndexT)*input_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_input, input, sizeof(InputT)*in_malloc_size, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaDeviceSynchronize());

    /*printf("the embedding table:\n");
    for (uint64_t i = 0; i < embedding_size; i ++) {
      for (int j = 0; j < embedding_dim; j ++)
        printf("%d ", h_embedding[i*emb_stride+j+emb_start_off]);
      printf("\n");
    }
    printf("the input table:\n");
    for (uint64_t i = 0; i < input_size; i ++) {
      for (int j = 0; j < input_dim; j ++)
        printf("%d ", h_input[i*in_stride+j+input_start_off]);
      printf("\n");
    }
    printf("the indices:\n");
    for (int i = 0; i < input_size; i ++)
      printf("%d ", h_indices[i]);
    printf("\n");*/

    int index_pos = 0;
    //int index_print_step = (input_size/100 > 1) ? input_size/100 : 1;
    bool valid = true;
    for (uint64_t i = 0; i < embedding_size; i ++) {
      if (i == h_indices[index_pos]) {
        //if (index_pos % index_print_step == 0)
          //printf("the %d th index for %d th iteration is %ld\n", index_pos, iter, i);
        if (h_embedding[i*emb_stride + emb_start_off] != in_malloc_size-input_start_off-1-index_pos*in_stride) {
          valid = false;
          printf("scattered, i = %lu, index_pos = %d, embedding ele is %d, should be %d\n",
                                i, index_pos, h_embedding[i*emb_stride + emb_start_off], 
                                in_malloc_size-input_start_off-1-index_pos*in_stride);
          break;
        }
        index_pos ++;
      } else {
        if (h_embedding[i*emb_stride + emb_start_off] != i*emb_stride) {
          valid = false;
          printf("original, i = %lu, embedding ele is %d, should be %lu\n",
                            i, h_embedding[i*emb_stride + emb_start_off], i*emb_stride);
          break;
        }
      }
    }
    if (valid)
      printf("    the %d th iteration passed the quick validation!\n", iter);
    else 
      printf("    the %d th iteration didn't pass!\n", iter);
#endif
#ifdef SCATTER
    thrust::sequence(thrust::device, embedding+emb_start_off, embedding+emb_malloc_size, 0);
#endif
    printf("\n");
  }
#ifdef QUICK_VALIDATION_SCATTER
  free(h_embedding);
  free(h_input);
#endif
#ifdef QUICK_VALIDATION_GATHER
  free(h_embedding);
  free(h_output);
#endif
  free(used);
  free(h_indices);
#ifdef SCATTER
  CUDA_TRY(cudaFree(input));
#endif
#ifdef GATHER
  CUDA_TRY(cudaFree(output));
#endif
  CUDA_TRY(cudaFree(embedding));
  CUDA_TRY(cudaFree(indices));
  printf("exit now\n");
  return 0;
}
