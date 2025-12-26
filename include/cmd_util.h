#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include "common.h"

void read_line_parameters (int argc, char *argv[], int myrank,
                           int *flag_b, int *flag_l, int *flag_x,
                           int *loop_count, int *buff_cycle, int *fix_buff_size ) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-l") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -l without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_l = 1;
            *loop_count = atoi(argv[i + 1]);
            if (*loop_count < 0) { // Can be 0 for endless mode (only for a2a and incast)
                fprintf(stderr, "Error: loop_count must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                    fprintf(stderr, "Error: specified -b without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_b = 1;
            *buff_cycle = atoi(argv[i + 1]);
            if (*buff_cycle <= 0) {
                fprintf(stderr, "Error: buff_cycle must be a positive integer.\n");
                exit(__LINE__);
            }
            i++;
        } else if (strcmp(argv[i], "-x") == 0) {
            if (i == argc) {
                if (myrank == 0) {
                fprintf(stderr, "Error: specified -x without a value.\n");
                }

                exit(__LINE__);
            }

            *flag_x = 1;
            *fix_buff_size = atoi(argv[i + 1]);
            if (*fix_buff_size < 0) {
                fprintf(stderr, "Error: fixed buff_size must be >= 0.\n");
                exit(__LINE__);
            }

            i++;
        } else {
            if (0 == myrank) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
            }

            exit(__LINE__);
        }
    }
}

typedef struct config
{
    // Debug parameeters
    int  verbose;

    // Repetitions and buffer size
    int loop_count;
    int buff_cycle;
    int fix_buff_size;

    // Process displacement
    int ppn;
    int npl;
    int tree_high;

    // Computed
    int nprocess;

} Config;

void parse_args(int argc, char ** argv, Config * config)
{
    config->verbose       = 0;

    config->loop_count    = 0;
    config->buff_cycle    = 0;
    config->fix_buff_size = 0;

    config->tree_high = 0;
    config->npl       = 1;
    config->ppn       = 1;

    int inc = 2;
    for (int i=1; i<argc; i+=inc)
    {
        inc = 2;
        const char * argname = (argv[i]);

        if (!strcmp(argname, "--loop-count"))
        {
            config->loop_count = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--fix-buff"))
        {
            config->fix_buff_size = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--buff-cycle"))
        {
            config->buff_cycle = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--verbose"))
        {
            config->verbose = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--tree-high"))
        {
            config->tree_high = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--nodes-per-leaf"))
        {
            config->npl = atoi(argv[i+1]);
        }
        else if (!strcmp(argname, "--process-per-node"))
        {
            config->ppn = atoi(argv[i+1]);
        }
    }

    if (config->loop_count == 0) {
        fprintf(stdout, "Error: parameeter --loop-count must be set grather than 0\n");
        exit(__LINE__);
    }
    if (config->buff_cycle == 0) {
        fprintf(stdout, "Error: parameeter --buff-cycle must be set grather than 0\n");
        exit(__LINE__);
    }

    config->nprocess = (config->ppn) * (config->npl) * (1<<(config->tree_high-1));

}

typedef struct my_mpi_comm
{
    MPI_Comm comm;
    int rank, size;
} MyMpiComm;

void init_mympicomm (MPI_Comm comm, MyMpiComm *mympicomm) {
    if (comm == MPI_COMM_NULL) {
        fprintf(stderr, "Error: %s found an MPI_COMM_NULL\n", __func__);
        exit(__LINE__);
    }

    mympicomm->comm = comm;
    MPI_Comm_size(mympicomm->comm, &(mympicomm->size));
    MPI_Comm_rank(mympicomm->comm, &(mympicomm->rank));
}

typedef struct mpi_comms
{
    MyMpiComm world;
    MyMpiComm node_comm;
    MyMpiComm cross_comm;

    int n_tree_comms;
    MyMpiComm *tree_comms;
    MyMpiComm *cross_comms;
} MpiComms;

void init_comms (Config * config, MpiComms * communicators) {

    init_mympicomm(MPI_COMM_WORLD, &(communicators->world));

    // Check correct process num
    if (communicators->world.size != config->nprocess) {
        fprintf(stderr, "Error: mismetch between defined processes (%d) and MPI processes (%d)\n", config->nprocess, communicators->world.size);
        exit(__LINE__);
    }

    MPI_Comm tmp_comm;
    int wrank = communicators->world.rank;
    int node_id  = wrank / (config->ppn);
    int cross_id = wrank % (config->ppn);

    MPI_Comm_split(MPI_COMM_WORLD, node_id,  wrank, &tmp_comm);
    init_mympicomm(tmp_comm, &(communicators->node_comm));
    MPI_Comm_set_name(tmp_comm, "NodeComm");

    MPI_Comm_split(MPI_COMM_WORLD, cross_id, wrank, &tmp_comm);
    init_mympicomm(tmp_comm, &(communicators->cross_comm));
    MPI_Comm_set_name(tmp_comm, "CrossNodeComm");

    int process_per_leaf = (config->ppn * config->npl);
    communicators->n_tree_comms = config->tree_high;
    communicators->tree_comms  = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(communicators->n_tree_comms)); // check > 0
    communicators->cross_comms = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(communicators->n_tree_comms)); // check > 0
    for (int i=0; i<communicators->n_tree_comms; i++) {
        int subtree_count   = 1<<i;

        int tree_comm_size  = process_per_leaf * subtree_count;
        int tree_comm_id    = wrank / tree_comm_size;
        MPI_Comm_split(MPI_COMM_WORLD, tree_comm_id, wrank, &tmp_comm);
        init_mympicomm(tmp_comm, &(communicators->tree_comms[i]));

        char comm_name[50];
        snprintf(comm_name, 50, "Level%dTreeComm", i);
        MPI_Comm_set_name(tmp_comm, comm_name);

        int cross_comm_id    = wrank % tree_comm_size;
        MPI_Comm_split(MPI_COMM_WORLD, cross_comm_id, wrank, &tmp_comm);
        init_mympicomm(tmp_comm, &(communicators->cross_comms[i]));

        snprintf(comm_name, 50, "CrossLevel%dComm", i);
        MPI_Comm_set_name(tmp_comm, comm_name);
    }
}

static void print_comm_info(const char *label, const MyMpiComm *c)
{
    char name[MPI_MAX_OBJECT_NAME];
    int name_len = 0;

    MPI_Comm_get_name(c->comm, name, &name_len);

    if (name_len == 0) {
        snprintf(name, MPI_MAX_OBJECT_NAME, "<unnamed>");
    }

    printf("[%-12s] name=%-20s rank=%4d size=%4d\n",
           label, name, c->rank, c->size);
}

void comms_info(const MpiComms *communicators)
{
    int world_rank = communicators->world.rank;

    /* Optional: make output readable by rank */
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < communicators->world.size; r++) {
        if (r == world_rank) {

            printf("\n=== Communicator info (world rank %d) ===\n", r);

            print_comm_info("WORLD",      &communicators->world);
            print_comm_info("NODE",       &communicators->node_comm);
            print_comm_info("CROSS-NODE", &communicators->cross_comm);

            for (int i = 0; i < communicators->n_tree_comms; i++) {
                char label[32];
                snprintf(label, sizeof(label), "TREE[%d]", i);
                print_comm_info(label, &communicators->tree_comms[i]);
            }

            for (int i = 0; i < communicators->n_tree_comms; i++) {
                char label[32];
                snprintf(label, sizeof(label), "CROSS[%d]", i);
                print_comm_info(label, &communicators->cross_comms[i]);
            }

            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#endif

