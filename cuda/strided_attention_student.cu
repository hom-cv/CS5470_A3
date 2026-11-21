#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <c10/cuda/CUDAException.h>

// CUDA kernel for the forward pass of strided attention
//
// T: data type (e.g., float, double, half)
//
// Computes: output = Softmax((Q * K_strided^T) / sqrt(head_dim)) * V_strided
//
// Grid/Block Dimensions:
// - gridDim.x: batch_size * num_heads
// - gridDim.y: seq_len (for queries)
// - blockDim.x: 256. Represents threads per query token.
//
template <typename T>
__global__ void strided_attention_forward_kernel(
    const T* __restrict__ q_ptr,
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ output_ptr,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int stride) {

    // =================================================================
    // TODO: IMPLEMENT YOUR STRIDED ATTENTION KERNEL HERE
    // =================================================================
    //
    // Note to self:
    // __shared__ = shared among threads in the same CUDA block
    // intiializing arrays
    __shared__ T query[256];
    __shared__ T weights[256];
    __shared__ T reduction[256];

    // Hints:
    // 1. Calculate thread/block indices to figure out which output element
    //    this thread block is responsible for.
    //    - `blockIdx.x` can map to (batch, head).
    //    - `blockIdx.y` can map to the query token index (`i`).
    //    - `threadIdx.x` is the thread within the block.
    const int tid = threadIdx.x;
    const int query_idx = blockIdx.y;
    const int num_keys = 1 + (seq_len - 1) / stride;
    const int64_t head_offset = static_cast<int64_t>(blockIdx.x) * seq_len * head_dim;
    const int64_t query_offset = head_offset + static_cast<int64_t>(query_idx) * head_dim;

    // 2. Load the query vector `q_i` for the current query token `i`.
    //    Since all threads in the block need it, you can load it into shared memory.
    if (tid < head_dim) {
        query[tid] = q_ptr[query_offset + tid];
    }
    __syncthreads();

    // 3. Compute dot products `Q_i * K_j^T` for all `j` in the stride.
    //    - Parallelize the dot product calculation across threads in the block.
    //    - Each thread can compute a partial dot product and then you can use
    //      a parallel reduction (e.g., using shared memory and `__syncthreads()`)
    //      to get the final score.
    // Each active thread computes a score for key 0, stride, 2*stride, ... .
    T score = -INFINITY;

    if (tid < num_keys) {
        const int64_t key_offset = head_offset + static_cast<int64_t>(tid) * stride * head_dim;
        T dot = 0;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * k_ptr[key_offset + d];
        }
        score = dot / sqrt(static_cast<T>(head_dim));
    }

    // 4. Compute Softmax.
    //    - Find the maximum score among the strided scores (for numerical stability). This is another parallel reduction.
    //    - Compute the exponential of each score (subtracting the max).
    //    - Compute the sum of the exponentials (another parallel reduction).
    //    - Normalize to get the final attention probabilities.
    // Reduce the maximum score to stabilize softmax. Inactive threads use -inf.
    reduction[tid] = score;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            reduction[tid] = reduction[tid] > reduction[tid + offset]
                ? reduction[tid] : reduction[tid + offset];
        }
        __syncthreads();
    }
    const T max_score = reduction[0];
    const T numerator = tid < num_keys ? exp(score - max_score) : T(0);
    // All threads must finish reading the maximum before reusing reduction.
    __syncthreads();

    reduction[tid] = numerator;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    weights[tid] = numerator / (reduction[0] + T(1e-6));
    __syncthreads();

    // 5. Compute the weighted sum of value vectors `V_j`.
    //    - Use the attention probabilities to weight the `v_j` vectors.
    //    - This is another parallel dot-product-like operation. Each thread can
    //      compute a part of the final output vector component.
    // Adjacent threads compute adjacent output features, sharing the weights.
    T value = 0;
    if (tid < head_dim) {
        for (int j = 0; j < num_keys; ++j) {
            const int64_t value_offset = head_offset + static_cast<int64_t>(j) * stride * head_dim;
            value += weights[j] * v_ptr[value_offset + tid];
        }
    }

    // 6. Write the final output vector to `output_ptr`.
    if (tid < head_dim) {
        output_ptr[query_offset + tid] = static_cast<T>(value);
    }
}


// C++ function that dispatches the CUDA kernel
torch::Tensor strided_attention_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int stride) {

    // Validate inputs
    TORCH_CHECK(q.is_cuda(), "Input tensor Q must be on a CUDA device");
    TORCH_CHECK(q.is_contiguous(), "Input tensor Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor V must be contiguous");

    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    TORCH_CHECK(stride > 0, "Stride must be positive");
    TORCH_CHECK(seq_len > 0 && head_dim > 0, "Sequence length and head dimension must be positive");
    TORCH_CHECK(head_dim <= 256 && 1 + (seq_len - 1) / stride <= 256,
                "This kernel supports head_dim and number of strided keys up to 256");

    // Create an output tensor of the same shape as Q
    auto output = torch::empty_like(q);

    // Define grid and block dimensions
    // Grid: One block per (batch, head, query_token)
    dim3 gridDim(batch_size * num_heads, seq_len);
    // Block: Threads to parallelize the work for a single query token
    dim3 blockDim(256);

    // Dispatch the kernel based on the data type
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "strided_attention_forward", ([&] {
        strided_attention_forward_kernel<scalar_t><<<gridDim, blockDim>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            stride
        );
    }));

    // Check for any CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}


// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("strided_attention_forward", &strided_attention_forward_cuda, "Strided Attention Forward (CUDA)");
}
