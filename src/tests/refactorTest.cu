#include "cmd_util.h"
#include "communicators.h"

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


int main(int argc, char ** argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    if (rank == 0) print_config(config);
    MPI_Barrier(MPI_COMM_WORLD);

    MpiComms *communicators = (MpiComms*)malloc(sizeof(MpiComms));
    communicators->init(config);

    if (communicators->world.rank == 0) {
        fprintf(stdout, "%d x %d x (tree high %d)\n", config->ppn, config->npl, config->tree_high);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    communicators->info();

    BlinkCommWrapper graph;
    graph.init(communicators);
    /*
    for (int i=0; i<size; i++) {
        if (i == rank) {
            fprintf(stdout, "--------------- Process %d ---------------\n", rank);
            graph.print(stdout);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        sleep(1);
    }
    */

    for (int i = 0; i < N_COMM_KIND; i++) {
        CommKind kind = static_cast<CommKind>(i);
        for (int level=0; level<communicators->n_tree_comms; level++) {
            graph.geninclist(kind, level);

            if (rank == 0) {
                fprintf(stdout, "--------------- CommKind %d level %d ---------------\n", kind, level);
                graph.print(stdout);
            }
            fflush(stdout);

            if (kind != TREE && kind != CROSS_TREE && kind != ROOT) break;
        }
    }

    bool test = communicators->check_node();
    if (rank == 0) fprintf(stdout, "Node check %s\n", (test) ? "true" : "false");

    test = communicators->check_addr();
    if (rank == 0) fprintf(stdout, "Addr check %s\n", (test) ? "true" : "false");

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
}
