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
        fprintf(stdout, "%d x %d x (tree high %d)\n", communicators->node_comm.size, communicators->group_comm.size, communicators->n_tree_comms);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stdout, "[%d] static_coordinates: [%d, %d]\n", rank, communicators->node_comm.rank, communicators->group_comm.rank);
    for (int i=0; i<communicators->n_tree_comms; i++) fprintf(stdout, "[%d] (%d: %d)\n", rank, i, communicators->tree_comms[i].rank);

    MPI_Finalize();
}
