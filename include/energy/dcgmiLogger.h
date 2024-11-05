#ifndef DCGMI_LOGGER_H_
#define DCGMI_LOGGER_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

int constexpr size_of_vector_ { 100000 };
// int constexpr nvml_device_name_buffer_size { 100 };


class dcgmiLogger {
  public:
    dcgmiLogger(std::string const &filename, int const &rank)
        : time_steps_{}, filename_{filename}, outfile_{}, loop_{false}, rank_{rank} {


        // Reserve memory for data
        time_steps_.reserve(size_of_vector_);

        // Open file
        if ( rank == 0 )
		outfile_.open(filename_, std::ios::out);

        // Print header
        // printHeader();
    }

    ~dcgmiLogger() {
        writeData();
    }

    void getStats() {
        loop_ = true;

        while (loop_) {

    // DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION = 156 #Total energy consumption for the GPU in mJ since the driver was last reloaded
    // DCGM_FI_DEV_POWER_USAGE         = 155 #Power usage for the device in Watts
    // DCGM_FI_DEV_POWER_USAGE_INSTANT = 157 #Current instantaneous power usage of the device in Watts
    // DCGM_FI_DEV_GPU_UTIL            = 203 #GPU Utilization
    // DCGM_FI_DEV_MEM_COPY_UTIL       = 204 #Memory Utilization
    //
    // ------------------------------------------- Occupancy ------------------------------------------
    // DCGM_FI_PROF_SM_OCCUPANCY                        = 1003 #The ratio of number of warps resident on an SM.
    //                                                         #(number of resident as a ratio of the theoretical
    //                                                         #maximum number of warps per elapsed cycle)
    //
    // ------------------------------------ ALU units utilizations ------------------------------------
    // DCGM_FI_PROF_SM_ACTIVE                           = 1002 		#The ratio of cycles an SM has at least 1 warp assigned
    //                                                         		# (computed from the number of cycles and elapsed cycles)
    // DCGM_FI_PROF_PIPE_TENSOR_ACTIVE 			= 1004
    // DCGM_FI_PROF_PIPE_FP64_ACTIVE 			= 1006
    // DCGM_FI_PROF_PIPE_FP32_ACTIVE 			= 1007
    // DCGM_FI_PROF_PIPE_FP16_ACTIVE 			= 1008
    // DCGM_FI_PROF_DRAM_ACTIVE                         = 1005 		#The ratio of cycles the device memory interface is active sending or receiving data.
    // DCGM_FI_DEV_CPU_UTIL_TOTAL         		= 1100 		# CPU Utilization, total
    //
    // -------------------------------- DataTransfer units utilizations -------------------------------
    // DCGM_FI_PROF_NVLINK_TX_BYTES			= 1011		# The total number of bytes of active NvLink tx (transmit) data including both
    // 									# header and payload. Per-link fields are available.
    // DCGM_FI_PROF_NVLINK_RX_BYTES			= 1012		# The total number of bytes of active NvLink rx (read) data including both
    // 									# header and payload. Per-link fields are available.
    // DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL 		= 449 		# Total bandwidth on the GPU's NVLink bridges
    // DCGM_FI_DEV_NVLINK_BANDWIDTH_LX 			= 440-445, 	# Bendwidth on a specific NVLink lane X with x in {0, ... , 17}
    // 							  475-480, 
    // 							  446-446, 494-496
    //
    // DCGM_FI_PROF_PCIE_TX_BYTES			= 1009		# The number of bytes of active PCIe tx (transmit) data including both header and payload. 
    // DCGM_FI_PROF_PCIE_RX_BYTES			= 1010		# The number of bytes of active PCIe rx (read) data including both header and payload.
    // # Note: both DCGM_FI_PROF_PCIE_TX_BYTES and DCGM_FI_PROF_PCIE_RX_BYTES are from the perspective of the GPU, so the first will reflect device to host (DtoH)
    // copies and the second the host to device (HtoD) ones.     
    //
    // -------------------------------- Resources and heat utilizations -------------------------------
    // DCGM_FI_DEV_GPU_UTIL                             = 203		# GPU Utilization.
    // DCGM_FI_DEV_GPU_TEMP 				= 150		# Current temperature readings for the device, in degrees C.
    // DCGM_FI_DEV_POWER_USAGE 				= 155 		# Power usage for the device in Watts.


            printf("Rank %d: Getting stats\n", rank_);
            // ----------- Orig -----------
	    //std::string command = "dcgmi dmon -e 1002,1003,1005,1009,1010,1011,1012 -c 3000 -d 1 -i " + std::to_string(rank_) + " | awk 'NR <= 10 && NR == 1 || NR > 10 && !/Entity|SMACT|ID/ {gsub(\" +\", \",\"); print}'";
	    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>
	    std::string command = "dcgmi dmon -e 1003,1002,1004,1006,1007,1008,1009,1010,1011,1012,203,150,155 -c 30000 -d 1" ;
	    // ---------------------------
            
            FILE* pipe = popen(command.c_str(), "r");
            if (pipe == nullptr) {
                std::cerr << "Error executing command: " << command << std::endl;
                return;
            }

            char buffer[1024];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                // Process the output of the command
                std::string output(buffer);
                time_steps_.push_back(output);
            }

            pclose(pipe);

            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void killThread() {
        // Retrieve a few empty samples
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Set loop to false to exit while loop
        loop_ = false;
    }

  private:
    std::vector<std::string> time_steps_;
    std::string filename_;
    std::ofstream outfile_;
    bool loop_;
    int rank_;


    void printHeader() {
        // // Print header
        // outfile_ << "dcgmi_output\n";
    }

    void writeData() {
        // Print data
        for (const auto& entry : time_steps_) {
            outfile_ << entry;
        }
        outfile_.close();
    }
};

#endif /* DCGMI_LOGGER_H_ */
