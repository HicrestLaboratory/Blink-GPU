#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <typeinfo>


namespace common {
    namespace logger {

        namespace fs = std::filesystem;
        // Utility struct to represent a single field in the CSV log
        // Used to build dynamic CSV rows with extra fields if needed
        struct CsvField {
            std::string key;
            std::string value;

            template<typename T>
            CsvField(std::string k, T&& v)
                    : key(std::move(k))
                    , value(to_string(std::forward<T>(v))) {}

                private:
                    static std::string to_string(const std::string& v) { return v; }
                    static std::string to_string(const char* v) { return v; }

                    template<typename T>
                    static std::string to_string(T v) {
                        std::ostringstream oss;
                        oss << v;
                        return oss.str();
                    }
                };

        class Logger {
        private:
        
            std::string output_file_path;
            std::string library_name;
            std::string collective_name;
            std::string gpu_mode = "gpu";
            
            int run_id;
            int mpi_rank;

            // Nuove variabili per informazioni sui nodi
            std::string hostname;
            int node_id;
            int total_nodes;
            bool is_multi_node;

            std::string get_timestamp() const {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                return ss.str();
            }
            
            std::string get_hostname_internal() const {
                char hostname_buf[256];
                if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0) {
                    return std::string(hostname_buf);
                }
                return "unknown";
            }
            
            void initialize_node_info() {
                // Ottieni hostname
                hostname = get_hostname_internal();

                // Ottieni informazioni MPI sui nodi
                int rank, size;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                mpi_rank = rank;
                MPI_Comm_size(MPI_COMM_WORLD, &size);
                
                // Raccogli tutti gli hostname dai vari rank
                const int max_hostname_len = 256;
                char local_hostname[max_hostname_len];
                strncpy(local_hostname, hostname.c_str(), max_hostname_len - 1);
                local_hostname[max_hostname_len - 1] = '\0';
                
                // Buffer per raccogliere tutti gli hostname
                std::vector<char> all_hostnames(size * max_hostname_len);
                
                // Gather di tutti gli hostname al rank 0
                MPI_Allgather(local_hostname, max_hostname_len, MPI_CHAR,
                            all_hostnames.data(), max_hostname_len, MPI_CHAR,
                            MPI_COMM_WORLD);
                
                // Analizza gli hostname per determinare nodi unici
                std::unordered_set<std::string> unique_hosts;
                std::vector<std::string> host_list;
                
                for (int i = 0; i < size; i++) {
                    std::string host(all_hostnames.data() + i * max_hostname_len);
                    host_list.push_back(host);
                    unique_hosts.insert(host);
                }
                
                // Determina total_nodes
                total_nodes = static_cast<int>(unique_hosts.size());
                
                // Determina is_multi_node
                is_multi_node = (total_nodes > 1);
                
                // Determina node_id (assegna un ID progressivo basato sull'ordine degli hostname unici)
                std::vector<std::string> sorted_hosts(unique_hosts.begin(), unique_hosts.end());
                std::sort(sorted_hosts.begin(), sorted_hosts.end());
                
                node_id = 0;
                for (int i = 0; i < sorted_hosts.size(); i++) {
                    if (sorted_hosts[i] == hostname) {
                        node_id = i;
                        break;
                    }
                }
                
                // Debug output (solo dal rank 0)
                if (rank == 0) {
                    std::cout << "[NODE INFO] Total nodes: " << total_nodes 
                            << ", Multi-node: " << (is_multi_node ? "yes" : "no") << std::endl;
                    std::cout << "[NODE INFO] Unique hostnames: ";
                    for (const auto& host : sorted_hosts) {
                        std::cout << host << " ";
                    }
                    std::cout << std::endl;
                }
            }

        
         
            
            void ensure_directory_exists() const {
                fs::path path(output_file_path);
                // Get the parent directory of the file
                fs::path dir = path.parent_path();

                if (!dir.empty() && !fs::exists(dir)) {
                    // Create directories recursively
                    if (fs::create_directories(dir)) {
                        std::cout << "Created path: " << dir << "\n";
                    } else {
                        std::cerr << "Failed to create path: " << dir << "\n";
                    }
                }
            }
            
            bool file_exists(const std::string& filename) const {
                return fs::exists(filename);
            }

            void write_header(std::ofstream& file, const std::vector<CsvField>& extras = {}) const {
                file << "library,collective,data_type,message_size_bytes,message_size_elements,num_ranks,global_rank,local_rank,hostname,node_id,total_nodes,is_multi_node,run_id,gpu_mode,test_passed,chain_size,total_time_ms,time_ms_1coll,total_device_energy_mj,device_energy_mj_1coll,total_host_energy_mj,host_energy_mj_1coll,goodput_Gb_per_s";
                for (const auto& f : extras)
                    file << "," << f.key;

                file << "\n";
            }

        public:
            Logger(const std::string& output_file_path, const std::string& library_name, const std::string& collective_name, const std::string& gpu_mode = "composite")
                : output_file_path(output_file_path), library_name(library_name), collective_name(collective_name), gpu_mode(gpu_mode) {
                ensure_directory_exists();
                initialize_node_info();
            }
            
             // T define the data type used for the collective operation
            template <typename T>
            struct ProfilingInfo {
                double time_ms; // time to solution of the rank to execute the specified collective in ms
                size_t message_size_bytes;
                double device_energy_mj; // in mj 
                double host_energy_mj; // in mj
                int global_rank;
                int local_rank;
                int num_ranks;
                int run_id; // index of the current run
                bool test_passed; // check the results of the collective operation
                int chain_size; // check the results of the collective operation
                std::string gpu_mode; // composite or flat: only for intel GPUs
            };

            // The log interface take as input ProfilingInfo but we can also add other fields to the csv as ExtraFields...
            template<typename T, typename... ExtraFields>
            void log_result(const ProfilingInfo<T>& info, ExtraFields&&... extras){
                /* 
                   For each message size we execute multiple times the same collective "c" so that we have a chain like this c_1, c_2, ... c_m, where m=chain_size.
                   The ProfilingInfo struct contains the time in ms and energy in mJ for the execution of the whole chain c_1, c_2, ... c_m,
                   To measure the time and energy of a single collective we devide time_ms and {device,host}_energy_mj with chain_size.
                */
               std::vector<CsvField> extra_fields{ std::forward<ExtraFields>(extras)... };

                double time_ms_1coll = info.time_ms / info.chain_size; 
                double device_energy_mj_1coll = info.device_energy_mj / info.chain_size; 
                double host_energy_mj_1coll = info.host_energy_mj / info.chain_size; 
                double data_Gb = static_cast<double>(info.message_size_bytes) / 1.25e+8;
                double goodput_Gb_per_s = data_Gb / (time_ms_1coll / 1000); // Gigabit per seconds 

                
                std::string filename = output_file_path;
                
                // Only rank 0 write the csv header 
                int needs_header = 0;
                if (info.global_rank == 0) {
                    bool is_new_file = !file_exists(filename);
                    needs_header = is_new_file ? 1 : 0;
                }

                MPI_Bcast(&needs_header, 1, MPI_INT, 0, MPI_COMM_WORLD);
                
                // Solo il rank 0 scrive l'header se necessario
                if (needs_header && info.global_rank == 0) {
                    std::ofstream header_file(filename, std::ios::app);
                    if (header_file.is_open()) {
                        write_header(header_file, extra_fields);
                        header_file.close();
                    }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                
                std::ofstream file(filename, std::ios::app);
                if (!file.is_open()) {
                    std::cerr << "Warning: Could not open log file: " << filename << std::endl;
                    return;
                }
                

              
                // Adde energy for each rank, then add also another line for avg time and energy for all the ranks
                // Scrivi la riga di dati
                file << library_name << ","
                    << collective_name << ","
                    << typeid(T).name() << ","
                    << info.message_size_bytes << ","
                    << info.message_size_bytes /  sizeof(T) << ","
                    << info.num_ranks << ","
                    << info.global_rank << "," // can be 0 to num_ranks or aggregate
                    << info.local_rank << "," // can be 0 to num_ranks or aggregate
                    << hostname << ","
                    << node_id << ","
                    << total_nodes << ","
                    << (is_multi_node ? "true" : "false") << ","
                    << info.run_id << "," 
                    << gpu_mode << "," // Only for Intel GPUs
                    << (info.test_passed ? "true" : "false") << ","
                    << info.chain_size << ","
                    << std::fixed << std::setprecision(3) << info.time_ms  << ","
                    << std::fixed << std::setprecision(3) << time_ms_1coll  << ","
                    << std::fixed << std::setprecision(3) << info.device_energy_mj  << ","
                    << std::fixed << std::setprecision(3) << device_energy_mj_1coll  << ","
                    << std::fixed << std::setprecision(3) << info.host_energy_mj  << ","
                    << std::fixed << std::setprecision(3) << host_energy_mj_1coll  << ","
                    << std::fixed << std::setprecision(3) << goodput_Gb_per_s;
                    
                    // Write extra fields if any
                    for (const auto& f : extra_fields)
                        file << "," << f.value;
                    
                    file << "\n";
                    // TODO: change header file
                    // TODO: Add tuple (power, timestamp) we can parse the tuple so that we have only the different power measurement so that we know that from timestamp x to y we have the same power
                   
                    
                file.close();


                 // Log anche su console per debug
                std::cout << "[LOG] " << library_name << " " << collective_name 
                        << " " << typeid(T).name() 
                        << " bytes=" << info.message_size_bytes 
                        << " size=" << info.message_size_bytes / sizeof(T) 
                        << " ranks=" << info.num_ranks 
                        << " rank=" << info.global_rank << " hostname=" << hostname << " node=" << node_id
                        << " run=" << info.run_id << " gpu_mode=" << info.gpu_mode << " passed=" << (info.test_passed ? "true" : "false")
                        << " time=" << time_ms_1coll << "ms" << " -> " << filename << std::endl;
            }
          
            
            
            // Metodi getter per le nuove informazioni sui nodi
            const std::string& get_hostname() const { return hostname; }
            int get_node_id() const { return node_id; }
            int get_total_nodes() const { return total_nodes; }
            bool get_is_multi_node() const { return is_multi_node; }
            
         
            
            static void print_usage() {
                std::cout << "\nLogger Usage:" << std::endl;
                std::cout << "  --output <path>  : Directory path for logging results (optional)" << std::endl;
                std::cout << "  If --output is not specified, results will only be printed to console" << std::endl;
                std::cout << "\nOutput format: CSV files with columns:" << std::endl;
                std::cout << "  timestamp, library, collective, data_type, message_size_bytes, message_size_elements, num_ranks, rank, hostname, node_id, total_nodes, is_multi_node, run_id, gpu_mode, test_passed, time_ms" << std::endl;
            }


        };
    } // end namespace logger
} // end namespace common