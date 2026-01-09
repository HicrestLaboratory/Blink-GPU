#include "cmd_util.h"
#include "communicators2.h"

char* simulate_addr_str(int rank, int size) {
    char *string = (char*)malloc(sizeof(char)*150);
    int l2, l1;
    int nodeid;

    l2 = (size >= 8) ? (rank/(size/2)) : 1 ;
    l1 = (size >= 8) ? (rank/(size/4)) : ((size >= 4) ? (rank/(size/2)) : 1) ;
    nodeid = rank % (size/8);
    sprintf(string, "HeadSwitche.isw-L2-CELLG%d.isw%d0%d0.lrdn%d%d0%d", l2, l2, l1, l2, l1, nodeid);

    return(string);
}

void print_vector_inline(const std::vector<char*>& v, const char* prefix=nullptr) {
    if (prefix != nullptr) printf("%s: ", prefix);
    for (char* s : v) {
        printf("%s ", s);
    }
    printf("\n");
}

template<typename T>
void print_array_inline(T* arr, int len, const char* prefix=nullptr) {
    if (prefix) std::cout << prefix << ": ";
    for (int i = 0; i < len; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

#include <tuple>
std::vector<std::tuple<const char*,int>> head = {{"HeadSwitche", 12}};
std::vector<std::tuple<const char*,int>> l2   = {{"isw-L2-0", 6}, {"isw-L2-1", 2}, {"isw-L2-2", 4}};
std::vector<std::tuple<const char*,int>> l1   = {{"isw-0", 1}, {"isw-1", 4}, {"isw-2", 1}, {"isw-3", 2}, {"isw-4", 3}, {"isw-5", 1}};

int* prefix_sum_counts(const std::vector<std::tuple<const char*, int>>& v, int alpha = 1) {
    int n = v.size();

    // Allocate n+1 elements
    int* prefix = (int*)malloc(sizeof(int) * (n + 1));

    prefix[0] = 0;
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + std::get<1>(v[i]);
    }

    for (int i = 0; i < n+1; ++i) prefix[i] *= alpha;
    return prefix;  // caller must free()
}

#include <stddef.h>

int search_prefix_sum(
    const int *prefix,
    int len,
    int x,
    int *diff)
{
    int lo = 0;
    int hi = len - 1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (prefix[mid] < x)
            lo = mid + 1;
        else
            hi = mid - 1;
    }

    // hi is now the last index such that prefix[hi] < x
    if (hi < 0)
        hi = 0;   // clamp (x <= prefix[0])

    int index = hi;
    *diff  = x - prefix[hi];
    return(index);
}


char* simulate_addr_str_2(int rank, int size) {
    if ((size % 12) != 0) {
        fprintf(stderr, "Error: nprocesses must be a multiple of 12\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    int processes_per_node = size/12;

    srand((unsigned int)(rank * 1337));

    int  x = rand();
    int *v = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&x, 1, MPI_INT, v, 1, MPI_INT, MPI_COMM_WORLD);

    int shuffled_rank = 0;
    for (int i = 0; i < size; i++) {
        if (v[i] < x)
            shuffled_rank++;
        else if (v[i] == x && i < rank)
            shuffled_rank++;
    }

    int *head_psum = prefix_sum_counts(head, processes_per_node);
    int *l2_psum   = prefix_sum_counts(l2,   processes_per_node);
    int *l1_psum   = prefix_sum_counts(l1,   processes_per_node);

    if (rank == 0) {
        print_array_inline(head_psum, head.size()+1, "head_psum");
        print_array_inline(l2_psum,     l2.size()+1,   "l2_psum");
        print_array_inline(l1_psum,     l1.size()+1,   "l1_psum");
    }

    int hdiff, myhead  = search_prefix_sum(head_psum, head.size(), rank, &hdiff);
    int l2diff, myl2   = search_prefix_sum(l2_psum,   l2.size(),   rank, &l2diff);
    int l1diff, myl1   = search_prefix_sum(l1_psum,   l1.size(),   rank, &l1diff);

    // if (rank == l1_psum[myl1+1]) fprintf(stdout, "!!!![%d]!!!! %d %s\n", rank, l1_psum[myl1+1], std::get<0>(l1[myl1]));
    // if (rank == l2_psum[myl2+1]) fprintf(stdout, "!!!![%d]!!!! %d %s\n", rank, l2_psum[myl2+1], std::get<0>(l2[myl2]));
    if (rank == l2_psum[myl2+1]) myl2++;
    if (rank == l1_psum[myl1+1]) myl1++;

    char *sim_addr = (char*)malloc(sizeof(char)*500);
    sprintf(sim_addr, "%s.%s.%s.N%d", std::get<0>(head[myhead]), std::get<0>(l2[myl2]), std::get<0>(l1[myl1]), (rank/processes_per_node)*10 );

    char **ghr_sim_addr = allgather_strings(strlen(sim_addr), sim_addr, MPI_COMM_WORLD);
    sim_addr = ghr_sim_addr[shuffled_rank];
    for (int i=0; i<size; i++) if (i!=shuffled_rank) free(ghr_sim_addr[i]);

    /* Print results in order */
    MPI_Barrier(MPI_COMM_WORLD);
    /*for (int r = 0; r < size; r++) {
        if (r == rank) {
            printf("Rank %d: %d -> shuffled_rank = %d, %s\n",
                   rank, x, shuffled_rank, sim_addr);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/


    return(sim_addr);
}


int main(int argc, char ** argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    // --------------------- Simulated ADDR ---------------------
    // I here generate simulated slurm addr style strings
    const char *addr_str = simulate_addr_str_2(rank, size);

    AddrStruct addr;
    addr.init(addr_str);
    for (int i=0; i<size; i++) {
        if (i == rank) {
            fprintf(stdout, "--------------- Process %d ---------------\n", rank);
            addr.print(stdout);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(250);
    }

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) fprintf(stdout, "-------------------------------------\n");

    ComputeNewWorld newworld_buffers;
    newworld_buffers.init(addr, MPI_COMM_WORLD);
    newworld_buffers.gen_all();


    sleep(1);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i=0; i<size; i++) {
        if (i == rank) {
            fprintf(stdout, "[%2d] ADDR string: %s (%d)\n", rank, addr.addr_str, rank);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(250);
    }

    sleep(1);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) fprintf(stdout, "     >>>>>>>>>>>>>>>>>>>>>>>     \n");

    int new_rank, new_size;
    MPI_Comm_size(newworld_buffers.new_world, &new_size);
    MPI_Comm_rank(newworld_buffers.new_world, &new_rank);
    for (int i=0; i<new_size; i++) {
        if (i == new_rank) {
            fprintf(stdout, "[%2d] ADDR string: %s (%d)\n", new_rank, addr.addr_str, rank);
        }
        fflush(stdout);
        MPI_Barrier(newworld_buffers.new_world);
        usleep(250);
    }

    if (rank == 0) fprintf(stdout, "-------------------------------------\n");

    sleep(1);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MpiComms2 newcomms;
    newcomms.init(newworld_buffers);
    newcomms.pregraph();
    newworld_buffers.clear();

    if(new_rank==0) newcomms.graph.shortPrint();
    if(new_rank==0) newcomms.graph.netPrint();

    sleep(1);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
