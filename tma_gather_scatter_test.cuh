#include <stdlib.h>
#include <stdio.h>
#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move
//#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
//TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

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
