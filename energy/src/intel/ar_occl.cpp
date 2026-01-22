
// You can use template paramter for specilize the funcition that run the collective with different data types
//TODO: Test multi nodo per vedere se cambia qualcosa dal punto di vista delle performance e dell'energy
#include <mpi.h>
#include "../utils/occl_utils.hpp"
#include "../../include/utils/ccl_data_types.hpp"  
#include <oneapi/ccl.hpp> // oneCCL main header

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "../../energy-profiler/include/profiler/power_profiler.hpp"
#include <vector>

#define MAX_RUN 5
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_S 10
#define POWER_SAMPLING_RATE_MS 5
#define MAX_BUF 100
#define MESSAGE_SIZE_FACTOR 16

namespace log = common::logger; // define logger namespace abbreviation
namespace data_types = common::utils::data_types; // define data_types namespace abbreviation
namespace prof_data_types = profiler::data_types; // define profiler::data_types namespace abbreviation
using Comm = data_types::CommWrapper<>::type;  // automaticall select ccl::communicator at compile time accroding to the defined macro USE_ONECCL

template<typename T>
void run(intel::utils::OneCCLContext& ctx, std::string& power_log_path){ 

    sycl::queue q = ctx.q; // sycl queue
    ccl::stream stream =std::move(*ctx.stream); // ccl stream
    Comm comm = std::move(*ctx.comm);
    log::Logger csv_log = std::move(*ctx.logger);

    int rank = ctx.global_rank;
    int numGPUs = ctx.local_rank_size;
    
    constexpr size_t ONE_GB = 1024 * 1024 * 1024;
    size_t *buff_size_byte = (size_t *) malloc(sizeof(size_t) * MAX_BUF);
    size_t num_elements=1;

    int i=0;

    while(num_elements * sizeof(T) <= ONE_GB ){
        buff_size_byte[i] = num_elements * sizeof(T);
        num_elements *= MESSAGE_SIZE_FACTOR;
        i++;
    }

    const int num_iters = i; // define the number of different message sizes to test in a single benchmark run

    // allocate device pointer
    T *d_sendbuf = sycl::malloc_device<T>(buff_size_byte[num_iters - 1], q); 
    T *d_recvbuf = sycl::malloc_device<T>(buff_size_byte[num_iters - 1], q); 

    /* init device side buffer */
    sycl::event init_event = q.submit([&](auto& h) {
        h.parallel_for(buff_size_byte[num_iters - 1] / sizeof(T), [=](auto id) {
            d_sendbuf[id] = rank + id + 1;
            d_recvbuf[id] = -1;
        });
    });

    // create dependencies vector: can be used in the collective to ensure that consecutive call to the collective are exectude in order
    std::vector<ccl::event> deps;
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();


    // allocate host pointer
    T *h_sendbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    T *h_recvbuf = (T *)malloc(buff_size_byte[num_iters - 1]);
    if (rank == 0) {
        std::cout<< "Warm up run for allreduce" << std::endl;    
    }
    
    for (int i = 0; i < WARM_UP_RUN; i++) {
        q.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1]).wait();

        auto start = std::chrono::high_resolution_clock::now();
        ccl::allreduce(d_sendbuf, d_recvbuf,  buff_size_byte[num_iters-1] / sizeof(T), ccl::reduction::sum, comm, stream, attr, deps).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        q.memcpy(d_sendbuf, h_sendbuf, buff_size_byte[0]).wait();       
    }

 

    auto mem_cpy_t_start = std::chrono::high_resolution_clock::now();
    q.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1]).wait();
    auto mem_cpy_t_end = std::chrono::high_resolution_clock::now();
    
    
    for (int i = 0; i < num_iters; i++) {
        int chain_size = 0; //  num of times that a collective is executed 
        int host_energy_counter=MAX_RUN;
        for (int run = 0; run < MAX_RUN; run++) {
            if (rank == 0) {
                std::cout<< "Run " << run << " for allreduce with message size "<< buff_size_byte[i] << " B" << std::endl;    
            }
            double ar_time_max = 0;
            chain_size = 0;
            
            profiler::PowerProfiler powerProf(rank % numGPUs, POWER_SAMPLING_RATE_MS);
            powerProf.start();
            int64_t ar_time_per_rank=0; // Store the time spent for each rank to complete the collective
            while (ar_time_max < (TIME_TO_ACHIEVE_S * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();
                
                ccl::allreduce(d_sendbuf, d_recvbuf,  buff_size_byte[i] / sizeof(T), ccl::reduction::sum, comm, stream, attr, deps).wait();

                auto end_s = std::chrono::high_resolution_clock::now();
                // Time to solution of the current rank
                int64_t single_run_us = std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                ar_time_per_rank+=single_run_us;

                // Time to solution to complete the collective 
                MPI_Allreduce(MPI_IN_PLACE, &single_run_us, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD); // Max time to solution
                ar_time_max += static_cast<double>(single_run_us) / 1000.0; // in ms 
                chain_size++;
            }
           
            powerProf.stop();
            
            prof_data_types::energy_t dev_energy_uj = powerProf.get_device_energy(); //device energy in uj for one collective run for the current rank
            double time_ms = static_cast<double>(ar_time_per_rank) / 1000.0; // time in ms for the current rank
            double dev_energy_mj = static_cast<double>(dev_energy_uj) / 1000.0; // device energy in mj for the current rank
            if (rank == 0) {
                std::cout << "[RESULT] Writing in logger file ..." << std::endl;
            }
            // host energy is 0 now
            log::Logger::ProfilingInfo<T> prof_info{time_ms, buff_size_byte[i], dev_energy_mj, 0.0, ctx.global_rank, ctx.local_rank, ctx.global_rank_size, run, true, chain_size, "composite"};  
            prof_data_types::power_trace_t power_trace = powerProf.get_power_execution_data(); // get power trace data and store it internally in the power_prof object
            std::string power_trace_str = prof_data_types::power_trace_to_string(power_trace);
            log::CsvField power_trace_field{"power_trace", power_trace_str};
            csv_log.log_result<T>(prof_info, power_trace_field); 
            
            //TODO: add support for host energy
            // double host_energy_mj = powerProf.get_host_energy() / static_cast<double>(chain_size); //host energy in mj for one collective run 

        }
    }
   
    q.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1]).wait();

    sycl::free(d_sendbuf, q);
    sycl::free(d_recvbuf, q);
    free(h_sendbuf);
    free(h_recvbuf);
}

int main(int argc, char *argv[]) {
    int rank, size;
    std::string power_log_path;
    std::string csv_log_path;
    
    if (argc != 3)
        return -1;
    else{   
        power_log_path = argv[1];
        csv_log_path = argv[2];
    }

    std::cerr << "[DEBUG] Power log path: " << power_log_path << std::endl;
    std::cerr << "[DEBUG] CSV log path: " << csv_log_path << std::endl;
    std::cerr << "[DEBUG] Initialize MPI with oneCCL ...: " << std::endl;

    intel::utils::OneCCLContext ctx = intel::utils::init_oneccl(csv_log_path,"a2a", intel::utils::GPUMode::Composite); // initialize oneCCL and MPI

    // Run with different data type
    // run<uint8_t>(comm, rank, numGPUs, power_log_path, csv_log_path);
    // run<int>(comm, rank, numGPUs, power_log_path, csv_log_path);
    run<float>(ctx, power_log_path);
    // run<double>(comm, rank, numGPUs, power_log_path, csv_log_path);

    
    MPI_Finalize();

    return 0;
}