/*
 * @Author: Jiaqi Gu (jqgu@utexas.edu)
 * @Date: 2020-10-22 12:18:52
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2020-10-22 13:01:43
 */
// auction_cuda.cu
#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <chrono>
#include "auction_cuda_kernel.cuh"


hr_clock_rep timer_start, timer_stop;

template <typename T>
void run_auction(
    int    num_graphs,
    int    num_nodes,
    T* h_data,      // data
    int*   h_person2item[], // results
    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,
    int num_runs,
    int verbose
)
{
    T *data;
    char* scratch;
    int* solutions;
    char* stop_flags;

    cudaMalloc((void **)&data,          BATCH_SIZE * num_nodes*num_nodes   * sizeof(T));
    cudaMalloc((void**) &scratch, num_graphs*(num_nodes*num_nodes)*sizeof(T));
    cudaMalloc((void**)& solutions, num_graphs*num_nodes*sizeof(int));
    cudaMalloc((void**)& stop_flags, sizeof(char) * num_graphs);

    cudaMemcpy(data, h_data, num_graphs* num_nodes*num_nodes* sizeof(T), cudaMemcpyHostToDevice);

    timer_start = get_globaltime();

    linearAssignmentAuctionCUDALauncher<T>(data,
                                solutions,
                                num_graphs,
                                num_nodes,
                                scratch,
                                stop_flags,
                                auction_max_eps,
                                auction_min_eps,
                                auction_factor,
                                MAX_ITERATIONS);

    cudaDeviceSynchronize();

    timer_stop = get_globaltime();


    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        cudaMemcpy(h_person2item[i], solutions + i * num_nodes, sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);
    }

    cudaFree(data);
    cudaFree(scratch);
    cudaFree(solutions);
    cudaFree(stop_flags);
    return;
}


template <typename T>
int load_data(T *raw_data, std::string & input_graph) {
    std::ifstream input_file(input_graph, std::ios_base::in);

    int i = 0;
    T val;
    while(input_file >> val) {
        raw_data[i] = val;
        i++;

    }
    return (int)sqrt(i);
}

int main(int argc, char **argv)
{
    std::string input_graph = "data/graph4";
    if(argc > 1)
    {
        input_graph = argv[1];
    }
    std::cerr << "loading:\t" << input_graph << std::endl;
    int num_nodes = NUM_NODES;
    int num_graphs = BATCH_SIZE;
    int *h_data = new int[num_graphs*num_nodes*num_nodes];
    int* h_person2item[BATCH_SIZE];


    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        num_nodes = load_data<int>(h_data + i*num_nodes*num_nodes, input_graph);
        h_person2item[i] = (int *)malloc(sizeof(int) * num_nodes);
    }

    int verbose = 1;

    run_auction<int>(
        num_graphs,
        num_nodes,
        h_data,
        h_person2item,
        AUCTION_MAX_EPS,
        AUCTION_MIN_EPS,
        AUCTION_FACTOR,
        NUM_RUNS,
        verbose
    );



    // // Print results
    for (int i = 0; i < 1; ++i)
    {
        std::cerr << "solution " << i << "\n";
        for (int j = 0; j < num_nodes; j++) {
            std::cerr << j << ":" << h_person2item[i][j] << ", ";
        }
        std::cerr << std::endl;

        float score = 0;
        for (int j = 0; j < num_nodes; j++) {
            score += h_data[i*num_nodes*num_nodes+j * num_nodes + h_person2item[i][j]];
        }

        std::cerr << "score=" << (int)score << std::endl;

    }
    delete[] h_data;
    printf("[D] run_auction takes %g ms\n", (timer_stop-timer_start)*get_timer_period());
}

#endif
