#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <chrono>

// --
// Define constants

#ifndef __RUN_VARS
#define __RUN_VARS
#define AUCTION_MAX_EPS 10.0 // Larger values mean solution is more approximate
#define AUCTION_MIN_EPS 1.0
#define AUCTION_FACTOR  0.1
#define NUM_RUNS        1
#define BATCH_SIZE     1024
#define MAX_ITERATIONS  500
#define NUM_NODES 128
#define BIG_NEGATIVE -9999999
#endif

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void)
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void)
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}


template <typename T>
__global__ void __launch_bounds__(1024, 16)
linearAssignmentAuctionKernel(const int num_nodes,
                                const T* __restrict__ cost_ptr,
                                int* solution_ptr,
                                T*  bids_ptr,
                                char* stop_flag_ptr,
                                const float auction_max_eps,
                                const float auction_min_eps,
                                const float auction_factor,
                                const int max_iterations)
{
    const int batch_id = blockIdx.x;
    const int node_id = threadIdx.x;
    __shared__ float auction_eps;
    __shared__ int num_iteration;
    __shared__ int num_assigned;

    extern __shared__ unsigned char s_data[];
    T* prices = (T*)s_data;
    int* sbids = (int*)(prices + num_nodes);
    int* person2item = sbids + num_nodes;

    int* item2person = person2item + num_nodes;

    if(node_id == 0){
        auction_eps = auction_max_eps;
        num_iteration = 0;
    }

    const T* __restrict__ data = cost_ptr + batch_id * num_nodes * num_nodes;
    int* solution_global = solution_ptr + batch_id * num_nodes;
    T* bids = bids_ptr + batch_id * num_nodes * num_nodes;
    char* stop_flag = stop_flag_ptr + batch_id;

    prices[node_id] = 0;

    __syncthreads();

    while(auction_eps >= auction_min_eps && num_iteration < max_iterations)
    {
        //clear num_assigned
        if(node_id == 0){
            num_assigned = 0;
        }

        //pre-init
        person2item[node_id] = -1;
        item2person[node_id] = -1;

        __syncthreads();
        //start iterative solving
        while(num_assigned < num_nodes && num_iteration < max_iterations)
        {
            //phase 1: init bid and bids

            for(int i = node_id; i < num_nodes*num_nodes; i += blockDim.x){
                bids[i] = 0;
            }
            sbids[node_id] = 0;

            __syncthreads();

            //phase 2: bidding
            if(person2item[node_id] == -1){
                T top1_val = BIG_NEGATIVE;
                T top2_val = BIG_NEGATIVE;
                int top1_col;
                T tmp_val;
                #pragma unroll 32
                for (int col = 0; col < num_nodes; col++)
                {
                    tmp_val = data[node_id * num_nodes + col];
                    if (tmp_val < 0)
                    {
                        continue;
                    }
                    tmp_val = tmp_val - prices[col];
                    if (tmp_val >= top1_val)
                    {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                    else if (tmp_val > top2_val)
                    {
                        top2_val = tmp_val;
                    }
                }
                if (top2_val == BIG_NEGATIVE)
                {
                    top2_val = top1_val;
                }
                T bid = top1_val - top2_val + auction_eps;

                atomicMax(sbids+top1_col, 1);
                bids[num_nodes * top1_col + node_id] = bid;

            }

            __syncthreads();

            //phase 3 : assignment
            if(sbids[node_id] != 0) {
                T high_bid  = 0;
                int high_bidder = -1;

                T tmp_bid = -1;
                #pragma unroll 64
                for(int i = 0; i < num_nodes; i++){
                    tmp_bid = bids[node_id * num_nodes + i];
                    if(tmp_bid > high_bid){
                        high_bid    = tmp_bid;
                        high_bidder = i;
                    }
                }

                int current_person = item2person[node_id];
                if(current_person >= 0){
                    person2item[current_person] = -1;
                } else {
                    atomicAdd(&num_assigned, 1);
                }

                prices[node_id]                += high_bid;
                person2item[high_bidder]          = node_id;
                item2person[node_id]              = high_bidder;
            }
            __syncthreads();

            //update iteration
            if(node_id == 0){
                num_iteration++;
            }
            __syncthreads();
        }
        //scale auction_eps
        if(node_id == 0){
            auction_eps *= auction_factor;
        }
        __syncthreads();
    }
    __syncthreads();
    //report whether finish solving
    if(node_id == 0){
        *stop_flag = (num_assigned == num_nodes);
    }
    //write result out

    solution_global[node_id] = person2item[node_id];

}

template <typename T>
void linearAssignmentAuction(
                const T* cost_matrics,
                int* solutions,
                const int num_graphs,
                const int num_nodes,
                char* scratch,
                char *stop_flags,
                float auction_max_eps,
                float auction_min_eps,
                float auction_factor,
                int max_iterations)
{
    //get pointers from scratch (size: num_nodes*num_nodes*sizeof(T))
    T* bids           = (T* )scratch;

    //launch solver
    linearAssignmentAuctionKernel<T><<<num_graphs, num_nodes, 4*num_nodes*sizeof(T)>>>
                                    (
                                        num_nodes,
                                        cost_matrics,
                                        solutions,
                                        bids,
                                        stop_flags,
                                        auction_max_eps,
                                        auction_min_eps,
                                        auction_factor,
                                        max_iterations
                                    );

    cudaDeviceSynchronize();

}

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

    linearAssignmentAuction<T>(data,
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
int load_data(T *raw_data) {
    std::ifstream input_file("./data/graph4", std::ios_base::in);

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

    std::cerr << "loading ./graph4" << std::endl;
    int num_nodes = NUM_NODES;
    int num_graphs = BATCH_SIZE;
    int *h_data = new int[num_graphs*num_nodes*num_nodes];
    int* h_person2item[BATCH_SIZE];


    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        num_nodes = load_data<int>(h_data + i*num_nodes*num_nodes);
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
