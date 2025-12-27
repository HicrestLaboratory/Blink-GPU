#include "../include/cmd_util.h"

int main(int argc, char ** argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Config * config = (Config *)(malloc(sizeof(Config)));
    parse_args(argc, argv, config);

    MpiComms *communicators = (MpiComms*)malloc(sizeof(MpiComms));
    init_comms(config, communicators);

    if (communicators->world.rank == 0) {
        fprintf(stdout, "%d x %d x (tree high %d)\n", config->ppn, config->npl, config->tree_high);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    comms_info(communicators);

    comm_graph graph;
    graph.init(communicators);
    for (int i=0; i<size; i++) {
        if (i == rank) {
            fprintf(stdout, "--------------- Process %d ---------------\n", rank);
            graph.print(stdout);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        sleep(1);
    }

    for (int i = 0; i < N_COMM_KIND; i++) {
        CommKind kind = static_cast<CommKind>(i);
        graph.geninclist(kind);

        if (rank == 0) {
            fprintf(stdout, "--------------- CommKind %d ---------------\n", kind);
            graph.print(stdout);
        }
        fflush(stdout);
    }

    MPI_Finalize();
}
