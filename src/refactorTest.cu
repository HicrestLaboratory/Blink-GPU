#include "../include/cmd_util.h"

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
    init_comms(config, communicators);

    if (communicators->world.rank == 0) {
        fprintf(stdout, "%d x %d x (tree high %d)\n", config->ppn, config->npl, config->tree_high);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    comms_info(communicators);

    comm_graph graph;
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

    bool test = check_node(communicators);
    if (rank == 0) fprintf(stdout, "Node check %s\n", (test) ? "true" : "false");

    test = check_addr(communicators);
    if (rank == 0) fprintf(stdout, "Addr check %s\n", (test) ? "true" : "false");

    MPI_Finalize();
}
