/*
 * @Author: Jiaqi Gu (jqgu@utexas.edu)
 * @Date: 2020-10-22 12:21:43
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2020-10-22 12:22:38
 */

#ifndef AUCTION_CUDA_KERNEL_CUH
#define AUCTION_CUDA_KERNEL_CUH
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
void linearAssignmentAuctionCUDALauncher(
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

#endif
