#include <stdlib.h>
#include <stdio.h>
#include <chrono>         
#include <algorithm>
#define MMULT_N 5
#define WEIGHT_MAX UINT32_MAX
#define TIME_LIMIT 3600

#define CHECK_ALLOC(p)                                                         \
{                                                                              \
    if (!(p)) {                                                                \
        printf("Out of Host memory!");                                         \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_WEIGHT(tot, c)                                                   \
{                                                                              \
    if (WEIGHT_MAX - tot < c) {                                                \
        printf("Total edge weight exceeding limit!\n");                        \
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

int read_graph(node_t** edge_start, node_t** edge_end, node_t** edge_weight, uint64_t* edge_n, node_t* node_n) {
    *node_n = read_uint64();
    *edge_n = read_uint64();
    CHECK_ALLOC( *edge_start = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_end = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_weight = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    node_t tot_weight = 0;
    for(uint64_t i=0; i<*edge_n; ++i) {
        (*edge_start)[i] = read_uint64(); 
        (*edge_weight)[i] = read_uint64(); 
        CHECK_WEIGHT(tot_weight, (*edge_weight)[i]);
        tot_weight += (*edge_weight)[i];
        (*edge_end)[i] = read_uint64(); 
    }
    return 0;
}

uint64_t randomize(uint64_t z) {
    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    z = (z ^ (z >> 31)) * 5;
    return ((z << 7) | (z >> (64 - 7))) * 9;
}

void spmv(node_t* edge_start, node_t* edge_end, node_t* edge_weight, uint64_t* w, uint64_t* z, uint64_t it, uint64_t edge_n) {
    for(uint64_t i=0; i<edge_n; ++i) {
        if(!i || edge_start[i] != edge_start[i-1]) {
            z[edge_start[i]] = 0;
        }
        z[edge_start[i]] += randomize(w[edge_end[i]] + it) * edge_weight[i];
    }
}

void make_indexes(node_t *indexes, node_t node_n) {
    for(node_t i = 0; i < node_n; ++i) indexes[i] = i;
}

void gather(uint64_t *s_z, uint64_t *z, node_t *indexes, node_t node_n) {
    for(node_t i = 0; i < node_n; ++i) s_z[i] = z[indexes[i]];
}

void sum(uint64_t *s_z, node_t node_n) {
    node_t cur = 0;
    uint64_t prec = s_z[0];
    s_z[0] = cur;
    for(node_t i = 1; i < node_n; ++i) {
        if(s_z[i] != prec) ++cur;
        prec = s_z[i];
        s_z[i] = cur;
    }
}

void reorder(uint64_t *w, uint64_t *s_z, node_t *indexes, node_t node_n) {
    for(node_t i = 0; i < node_n; ++i) w[indexes[i]] = s_z[i];
}

int main(void) {
    node_t start_node_n = 0, node_n = 0, new_node_n = 0, *edge_start, *edge_end, *edge_weight, *indexes;
    uint64_t edge_n = 0, it = 0, *swp, *w, *z, *s_z;

    CHECK_RESULT( read_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    start_node_n = node_n;
    new_node_n = node_n;

    CHECK_ALLOC( w = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( s_z = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( indexes = (node_t*)malloc(sizeof(node_t) * node_n) );

    auto st = std::chrono::steady_clock::now();

    for(node_t i = 0; i<node_n; ++i){
        w[i] = 1;
        z[i] = 0;
    }

    do {
        node_n = new_node_n;

        spmv(edge_start, edge_end, edge_weight, w, z, it, edge_n);

        make_indexes(indexes, start_node_n);

        std::sort(indexes, indexes + start_node_n, [&](int i, int k) {
            return z[i] < z[k] || (z[i] == z[k] && w[i] < w[k]);
        });

        gather(s_z, z, indexes, start_node_n);

        sum(s_z, start_node_n);

        new_node_n = s_z[start_node_n - 1] + 1;

        reorder(w, s_z, indexes, start_node_n);

        ++it;

    } while(node_n != new_node_n);

    auto en = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / 1000000.0;
    printf("%f\n", time_s);
    printf("1\n");
    printf("%u\n", new_node_n);

    return 0;
}

