#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <omp.h>
#include <chrono>         
#define THREAD_N 256
#define MMULT_N 5
#define NOT_INIT UINT64_MAX
#define OFF_BATCH 0
#define IN_BATCH 1
#define DANGLING 2

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA_LOG(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_ALLOC(p)                                                         \
{                                                                              \
    if (!(p)) {                                                                \
        printf("Out of Host memory!\n");                                       \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_RESULT(r)                                                        \
{                                                                              \
    if (r) {                                                                   \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

typedef uint32_t node_t;

__global__ void init_partition(node_t node_n, uint8_t* batch_mask,  uint64_t* z) { 
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        uint8_t status = batch_mask[i];
        if(status == IN_BATCH) {
            z[i] = node_n + 1;
        } else if(status == DANGLING) {
            z[i] = 0;
        } else {
            z[i] = i + 1;
        }
    }
}

__device__ static inline uint64_t randomize(uint64_t v) {
    v = v + 0x9e3779b97f4a7c15;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9;
    v = (v ^ (v >> 27)) * 0x94d049bb133111eb;
    v = (v ^ (v >> 31)) * 5;
    return ((v << 7) | (v >> (64 - 7))) * 9;
}

__global__ void randomize_w(uint64_t *z, uint64_t *w, uint64_t it, node_t node_n) { 
    node_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        w[i] = randomize(z[i] + it);
    }
}

__global__ void set_values(uint64_t edge_n, uint64_t* edge_weight, node_t* edge_end, uint64_t* values, uint64_t* w, uint64_t it) { 
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < edge_n) {
        uint64_t end = edge_end[i];
        uint64_t weight = edge_weight[i];
        values[i] = randomize(w[end] + it) * weight;
    }
}

template<class T>
__global__ void count_unique_elements(uint64_t n, node_t* count, T* v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(i< n && (!i || v[i] != v[i - 1])) atomicAdd(count, 1);
}

template<class T>
__global__ void create_index(T* indexes, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        indexes[i] = i;
    }
}

__global__ void mend_partition(uint64_t* z, uint64_t* sorted_z, uint8_t *batch_mask, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        uint8_t status = batch_mask[i];
        if(status == OFF_BATCH) {
            z[i] = UINT64_MAX;
        } else if(status == DANGLING) {
            z[i] = 0;
        } else {
            z[i] = sorted_z[i];
        }
    }
}

__global__ void compute_thruths_lex(uint64_t* z, uint64_t* w, node_t *indexes, node_t* truths, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        truths[i] = i && (z[i-1] != z[i] || w[indexes[i-1]] != w[indexes[i]]);
    }
}

__global__ void compute_thruths(uint64_t* z, node_t* truths, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        truths[i] = i && z[i-1] != z[i];
    }
}

template<class T>
__global__ void reorder(node_t* indexes, T* shuffled, uint64_t* z, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        z[indexes[i]] = shuffled[i];
    }
}

template<class T>
__global__ void gather(node_t* indexes, T* shuffled, uint64_t* z, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        z[i] = shuffled[indexes[i]];
    }
}

__global__ void reorder_pairs(node_t* indexes, node_t* cumsum, uint64_t* z, uint64_t* w, node_t node_n) {
    node_t i = (node_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        node_t index = indexes[i];
        z[index] = ((uint64_t)cumsum[i] << 32) | ((uint32_t)w[index]);
    }
}

uint64_t read_uint64() {
    char ch = getchar();
    uint64_t n = 0;
    uint64_t c = 0;
    while(ch != ' ' && ch != '\n') {
        c = ch - '0';
        n = (n*10) + c;
        ch = getchar();
    }
    return n;
}

int read_graph(node_t** edge_start, node_t** edge_end, uint64_t** edge_weight, uint64_t* edge_n, node_t* node_n) {
    *node_n = read_uint64();
    *edge_n = read_uint64();
    CHECK_ALLOC( *edge_start = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_end = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_weight = (uint64_t*)malloc(*edge_n * sizeof(uint64_t)) );
    for(uint64_t i=0; i<*edge_n; ++i) {
        (*edge_start)[i] = read_uint64();
        (*edge_weight)[i] = read_uint64();
        (*edge_end)[i] = read_uint64();
    }
    return 0;
}

int main(int argc, char* argv[]) {
    uint8_t *batch_mask, **d_batch_mask;

    uint64_t **d_w, **d_z, **d_edge_weight, edge_n, new_edge_n, batches, max_batch_edge_n, max_gpus,
             *considered_nodes, *weight_indexes, *batch_partition;

    node_t node_n, new_node_n, *w, *edge_start, *edge_end,
           **d_edge_start, **d_edge_end, **d_unique_node_count,
           **d_z_unique_n, *label_map;

    uint64_t *edge_weight;

    void **d_tmp, **d_buffer;

    size_t *tmp_size;

    auto reduction_op = cuda::std::plus{};

    CHECK_RESULT( read_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    new_node_n = node_n;
    new_edge_n = edge_n;

    max_batch_edge_n = argc >= 2 ? atoll(argv[1]) : edge_n;
    max_gpus = argc == 3 ? atoll(argv[2]) : 1;

    CHECK_ALLOC( w = (node_t*)malloc(sizeof(node_t) * node_n) );
    CHECK_ALLOC( batch_mask = (uint8_t*)malloc(sizeof(uint8_t) * node_n * max_gpus) );
    CHECK_ALLOC( considered_nodes = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( weight_indexes = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( batch_partition = (uint64_t*)malloc(sizeof(uint64_t) * node_n * max_gpus) );
    CHECK_ALLOC( label_map = (node_t*)malloc(sizeof(node_t) * node_n * max_gpus) );

    CHECK_ALLOC( d_batch_mask = (uint8_t**)malloc(max_gpus * sizeof(uint8_t*)) );
    CHECK_ALLOC( d_unique_node_count = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_z_unique_n = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_buffer = (void**)malloc(max_gpus * sizeof(void*)) );
    CHECK_ALLOC( d_z = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_w = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_edge_weight = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_edge_start = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_edge_end = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_tmp = (void**)malloc(max_gpus * sizeof(void*)) );
    CHECK_ALLOC( tmp_size = (size_t*)malloc(max_gpus * sizeof(size_t)) );

    size_t KEYS_SIZE = node_n * sizeof(node_t);
    size_t PARTITION_SIZE = node_n * sizeof(uint64_t);
    size_t WEIGHTS_SIZE = max_batch_edge_n * sizeof(uint64_t);
    size_t EDGE_COORDS_SIZE = max_batch_edge_n * sizeof(uint32_t);
    size_t MUL_VALUES_SIZE = WEIGHTS_SIZE;
    size_t BATCH_MASK_SIZE = node_n * sizeof(uint8_t);
    size_t SCRATCHPAD_SIZE = max(MUL_VALUES_SIZE, PARTITION_SIZE) + 4 * KEYS_SIZE;

    for(uint64_t g = 0; g < max_gpus; ++g) {
        cudaSetDevice(g);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        CHECK_CUDA( cudaMalloc((void **)&d_unique_node_count[g], sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_z_unique_n[g], sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_buffer[g], SCRATCHPAD_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_z[g], PARTITION_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_w[g], PARTITION_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_weight[g], WEIGHTS_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_start[g], EDGE_COORDS_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_end[g], EDGE_COORDS_SIZE) );
        CHECK_CUDA( cudaMalloc((void **)&d_batch_mask[g], BATCH_MASK_SIZE) );

        size_t tmp_sizes_bytes[4] = {0};

        uint64_t *d_cv = (uint64_t*)d_buffer[g];
        uint64_t *d_sz = (uint64_t*)d_buffer[g];
        node_t *d_idx = (node_t*)(d_cv + max(max_batch_edge_n, (uint64_t)node_n));
        node_t *d_sidx = d_idx + node_n;
        node_t *d_tr = d_sidx + node_n;
        node_t *d_cs = d_tr + node_n;

        cub::DeviceReduce::ReduceByKey(nullptr, tmp_sizes_bytes[0], d_edge_start[g], d_idx, d_cv, d_z[g], d_unique_node_count[g], reduction_op, max_batch_edge_n, stream);
        cub::DeviceRadixSort::SortPairs(nullptr, tmp_sizes_bytes[1], d_z[g], d_sz, d_idx, d_sidx, node_n, 0, sizeof(uint64_t) * 8, stream);
        cub::DeviceScan::InclusiveSum(nullptr, tmp_sizes_bytes[2], d_tr, d_cs, node_n, stream);
        cub::DeviceRadixSort::SortKeys(nullptr, tmp_sizes_bytes[3], d_w[g], d_sz, node_n, 0, sizeof(uint64_t) * 8, stream);

        CHECK_CUDA( cudaStreamSynchronize(stream) );

        tmp_size[g] = 0;
        for(size_t s : tmp_sizes_bytes) tmp_size[g] = max(s, tmp_size[g]);

        CHECK_CUDA( cudaMalloc(&d_tmp[g], tmp_size[g]) );
        cudaStreamDestroy(stream);
    }

    auto st = std::chrono::steady_clock::now();

    do {
        node_n = new_node_n;
        edge_n = new_edge_n;
        new_node_n = 0;
        batches = ceil((double)edge_n / max_batch_edge_n);

        for(node_t i = 0; i < node_n; ++i) w[i] = node_n;

        uint64_t gpu, batch, batch_start, batch_end, new_batch_start, new_batch_end, batch_edge_n, i, *current_batch_partition, it;
        cudaStream_t stream;
        uint8_t *current_batch_mask;
        node_t unique_node_count, batch_node_n, off_batch_node_n, w_unique_n, z_unique_n, new_z_unique_n;
        uint64_t *d_swp_local;
        node_t *current_label_map;

        #pragma omp parallel num_threads(max_gpus) private(gpu, stream, batch, batch_start, batch_end, batch_edge_n, i, current_batch_mask, unique_node_count, z_unique_n, new_z_unique_n, w_unique_n, batch_node_n, d_swp_local, current_batch_partition, current_label_map, it)
        {
            gpu = omp_get_thread_num();
            CHECK_CUDA_LOG( cudaSetDevice(gpu) );
            CHECK_CUDA_LOG( cudaStreamCreate(&stream) );

            uint64_t *d_computed_values = (uint64_t*)d_buffer[gpu];
            uint64_t *d_sorted_z = (uint64_t*)d_buffer[gpu];
            node_t *d_indexes = (node_t*)(d_computed_values + max(max_batch_edge_n, (uint64_t)node_n));
            node_t *d_sorted_indexes = d_indexes + node_n;
            node_t *d_truths = d_sorted_indexes + node_n;
            node_t *d_cumsum = d_truths + node_n;

            #pragma omp for
            for(batch = 0; batch < batches; ++batch) {
                batch_start = batch * max_batch_edge_n;
                batch_end = min(edge_n, batch_start + max_batch_edge_n);
                batch_node_n = 0;

                current_batch_mask = batch_mask + (node_n * gpu);

                for(i = 0; i < node_n; ++i) current_batch_mask[i] = batch == 0 ? DANGLING : OFF_BATCH;

                new_batch_start = NOT_INIT;
                new_batch_end = NOT_INIT;

                for(i = batch_start; i < batch_end; ++i) {
                    if((!i || edge_start[i-1] != edge_start[i]) && new_batch_start == NOT_INIT) new_batch_start = i;

                    if(i && new_batch_start != NOT_INIT && edge_start[i-1] != edge_start[i]) new_batch_end = i;

                    current_batch_mask[edge_start[i]] = IN_BATCH;
                }

                if(batch_end == edge_n || edge_start[batch_end] != edge_start[batch_end - 1]) new_batch_end = batch_end;

                batch_start = new_batch_start;
                batch_end = new_batch_end;

                if(batch_start == NOT_INIT || batch_end == NOT_INIT) continue;

                batch_edge_n = batch_end - batch_start;

                for(i = 0; i < batch_start; ++i) current_batch_mask[edge_start[i]] = OFF_BATCH;

                for(i = batch_end; i < edge_n; ++i) current_batch_mask[edge_start[i]] = OFF_BATCH;

                for(i = 0; i < node_n; ++i) if(current_batch_mask[i]) ++batch_node_n;

                off_batch_node_n = node_n - batch_node_n;
                z_unique_n = new_z_unique_n = off_batch_node_n + 1;

                cudaMemcpyAsync(d_batch_mask[gpu], current_batch_mask, node_n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_edge_start[gpu], edge_start + batch_start, batch_edge_n * sizeof(node_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_edge_end[gpu], edge_end + batch_start, batch_edge_n * sizeof(node_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_edge_weight[gpu], edge_weight + batch_start, batch_edge_n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                cudaMemsetAsync(d_unique_node_count[gpu], 0, sizeof(node_t), stream);

                count_unique_elements<<<(batch_edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(batch_edge_n, d_unique_node_count[gpu], d_edge_start[gpu]);

                cudaMemcpyAsync(&unique_node_count, d_unique_node_count[gpu], sizeof(node_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                init_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(node_n, d_batch_mask[gpu], d_w[gpu]);
                it = 0;
                do {
                    z_unique_n = new_z_unique_n;

                    set_values<<<(batch_edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(
                        batch_edge_n,
                        d_edge_weight[gpu],
                        d_edge_end[gpu],
                        d_computed_values,
                        d_w[gpu],
                        it
                    );

                    cub::DeviceReduce::ReduceByKey(
                        d_tmp[gpu],
                        tmp_size[gpu],
                        d_edge_start[gpu],
                        d_indexes,
                        d_computed_values,
                        d_z[gpu],
                        d_unique_node_count[gpu],
                        reduction_op,
                        batch_edge_n,
                        stream
                    );

                    reorder<<<(unique_node_count + (THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(
                            d_indexes, d_z[gpu], d_sorted_z, unique_node_count);

                    mend_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_z[gpu], d_sorted_z, d_batch_mask[gpu], node_n);
                    
                    create_index<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_indexes, node_n);

                    cub::DeviceRadixSort::SortPairs(d_tmp[gpu], tmp_size[gpu], d_w[gpu], d_sorted_z, d_indexes, d_sorted_indexes, node_n, 0, sizeof(node_t) * 8, stream);

                    gather<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_sorted_indexes, d_z[gpu], d_sorted_z, node_n);

                    cub::DeviceRadixSort::SortPairs(d_tmp[gpu], tmp_size[gpu], d_sorted_z, d_z[gpu], d_sorted_indexes, d_indexes, node_n, 0, sizeof(uint64_t) * 8, stream);

                    compute_thruths_lex<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_z[gpu], d_w[gpu], d_indexes, d_truths, node_n);

                    cub::DeviceScan::InclusiveSum(d_tmp[gpu], tmp_size[gpu], d_truths, d_cumsum, node_n, stream);

                    reorder<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_indexes, d_cumsum, d_z[gpu], node_n);

                    cudaMemcpyAsync(&new_z_unique_n, d_cumsum + node_n - 1, sizeof(node_t), cudaMemcpyDeviceToHost, stream);

                    cudaStreamSynchronize(stream);

                    ++new_z_unique_n;

                    do {
                        ++it;

                        randomize_w<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(d_z[gpu], d_w[gpu], it, node_n);

                        cub::DeviceRadixSort::SortKeys(d_tmp[gpu], tmp_size[gpu], d_w[gpu], d_sorted_z, node_n, 0, sizeof(uint64_t) * 8, stream);

                        cudaMemsetAsync(d_z_unique_n[gpu], 0, sizeof(node_t), stream);

                        count_unique_elements<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(node_n, d_z_unique_n[gpu], d_sorted_z);

                        cudaMemcpyAsync(&w_unique_n, d_z_unique_n[gpu], sizeof(node_t), cudaMemcpyDeviceToHost, stream);

                        cudaStreamSynchronize(stream);

                    } while(new_z_unique_n != w_unique_n);

                    d_swp_local = d_z[gpu];
                    d_z[gpu] = d_w[gpu];
                    d_w[gpu] = d_swp_local;

                } while(z_unique_n != new_z_unique_n);

                current_batch_partition = batch_partition + node_n * gpu;

                cudaMemcpy(current_batch_partition, d_w[gpu], node_n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

                current_label_map = label_map + node_n * gpu;
                for(i = 0; i < node_n; ++i) current_label_map[i] = node_n;

                for(i = 0; i < node_n; ++i) {
                    if(current_batch_mask[i]) {
                        node_t local_label = current_batch_partition[i];
                        if(current_label_map[local_label] == node_n) {
                            #pragma omp critical
                            {
                                current_label_map[local_label] = new_node_n++;
                            }
                        }
                        w[i] = current_label_map[local_label];
                    }
                }
            }

            cudaStreamDestroy(stream);
        }

        if(batches > 1) {
            new_edge_n = 0;
            uint64_t first_edge_i;
            node_t current_node;
            uint8_t counting;

            for(node_t i = 0; i<node_n; ++i) {
                weight_indexes[i] = edge_n;
                considered_nodes[i] = 0;
                if(w[i] == node_n) w[i] = new_node_n++;
            }

            for(uint64_t i=0; i<edge_n; ++i) {
                if(!i || edge_start[i] != current_node) {
                    current_node = edge_start[i];
                    counting = !considered_nodes[w[current_node]];
                    first_edge_i = new_edge_n;
                    considered_nodes[w[edge_start[i]]] = 1;
                }

                if(counting) {
                    edge_start[i] = w[edge_start[i]];
                    edge_end[i] = w[edge_end[i]];

                    if(weight_indexes[edge_end[i]] == edge_n || weight_indexes[edge_end[i]] < first_edge_i) {
                        edge_start[new_edge_n] = edge_start[i];
                        edge_end[new_edge_n] = edge_end[i];
                        edge_weight[new_edge_n] = edge_weight[i];
                        weight_indexes[edge_end[i]] = new_edge_n;
                        ++new_edge_n;
                    } else {
                        edge_weight[weight_indexes[edge_end[i]]] += edge_weight[i];
                    }
                }
            }
        }
    } while(batches > 1 && batches > ceil(new_edge_n / (float)max_batch_edge_n) && node_n > new_node_n);

    auto en = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / 1000000.0;

    printf("%lf\n", time_s);
    printf("%u\n", batches == 1);
    printf("%lu\n", new_node_n);

    return 0;
}
