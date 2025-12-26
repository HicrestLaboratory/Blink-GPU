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

    MPI_Finalize();
}
