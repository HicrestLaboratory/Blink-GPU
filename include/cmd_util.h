#ifndef TEST_UTILS_CUH
#define TEST_UTILS_CUH

#include "common.h"
#include <functional>
#include <string>
#include <vector>
#include <iostream>

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

typedef struct process_env {

    char* home = nullptr;
    char* slurm_addr = nullptr;
    char* slurm_node = nullptr;

    /*
    // parsed copies / containers (C++ types owned by this struct)
    std::string slurm_hostname;
    std::vector<std::string> switch_names;

    // Parse slurm_addr (copy to std::string locally), fill slurm_hostname
    // and switch_names (in reverse order of the dotted fields, excluding hostname).
    void splitSlurmAddr() {
        switch_names.clear();
        slurm_hostname.clear();

        if (!slurm_addr) return;

        // copy to std::string for safe parsing
        std::string s{slurm_addr};

        if (s.empty()) return;

        std::vector<std::string> parts;
        size_t start = 0;
        while (true) {
            size_t pos = s.find('.', start);
            if (pos == std::string::npos) {
                parts.push_back(s.substr(start));
                break;
            }
            parts.push_back(s.substr(start, pos - start));
            start = pos + 1;
        }

        if (parts.empty()) return;

        // last part is hostname
        slurm_hostname = parts.back();

        // remaining parts (all except last) go into switch_names in reverse order
        for (int i = static_cast<int>(parts.size()) - 2; i >= 0; --i) {
            switch_names.push_back(parts[i]);
        }
    }
    */

    // Initialize pointers from environment; slurm_addr remains a char*
    // (points into environment memory - do NOT free it).
    void init_processenv(void) {
        home = std::getenv("HOME");
        slurm_node = std::getenv("SLURM_NODEID");
        slurm_addr = std::getenv("SLURM_TOPOLOGY_ADDR");

        // if (slurm_addr != nullptr) splitSlurmAddr();
    }

} ProcessEnv;

typedef struct mpi_comms
{
    MyMpiComm world;
    MyMpiComm node_comm;
    MyMpiComm cross_comm;

    int n_tree_comms;
    MyMpiComm *tree_comms;
    MyMpiComm *cross_comms;

    ProcessEnv myenv;
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

    communicators->myenv.init_processenv();
}

static void print_comm_info(const char *label, const MyMpiComm *c, FILE *fp)
{
    char name[MPI_MAX_OBJECT_NAME];
    int name_len = 0;

    MPI_Comm_get_name(c->comm, name, &name_len);

    if (name_len == 0) {
        snprintf(name, MPI_MAX_OBJECT_NAME, "<unnamed>");
    }

    int  hostname_len = 0;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(host_name, &hostname_len);

    if (hostname_len == 0) {
        snprintf(host_name, MPI_MAX_PROCESSOR_NAME, "<unnamed>");
    }

    fprintf(fp, "[%-12s] hostname=%-20s commname=%-20s rank=%4d size=%4d\n",
           label, host_name, name, c->rank, c->size);
}

void comms_info(const MpiComms *communicators, FILE *fp=stdout) {
    int world_rank = communicators->world.rank;

    if (world_rank == 0) {
        int len;
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        MPI_Get_library_version(version, &len);
        printf("%s\n", version);
    }

    /* Optional: make output readable by rank */
    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < communicators->world.size; r++) {
        if (r == world_rank) {

            fprintf(fp, "\n=== Communicator info (world rank %d) ===\n", r);

            print_comm_info("WORLD",      &communicators->world, fp);
            print_comm_info("NODE",       &communicators->node_comm, fp);
            print_comm_info("CROSS-NODE", &communicators->cross_comm, fp);

            for (int i = 0; i < communicators->n_tree_comms; i++) {
                char label[32];
                snprintf(label, sizeof(label), "TREE[%d]", i);
                print_comm_info(label, &communicators->tree_comms[i], fp);
            }

            for (int i = 0; i < communicators->n_tree_comms; i++) {
                char label[32];
                snprintf(label, sizeof(label), "CROSS[%d]", i);
                print_comm_info(label, &communicators->cross_comms[i], fp);
            }

            fprintf(fp, "[%-12s] HOME=%-20s\n",
                   "ENV-VARS", communicators->myenv.home);

            fprintf(fp, "[%-12s] SLURM_NODEID=%-5s, SLURM_TOPOLOGY_ADDR=%-20s\n",
                "SLURM-ENV", communicators->myenv.slurm_node, communicators->myenv.slurm_addr);
            fflush(fp);

            // if (communicators->myenv.slurm_addr != nullptr) {
            //     printf("slurm_hostname: %s\n", communicators->myenv.slurm_hostname.c_str());
            //     for (auto &p : communicators->myenv.switch_names) std::cout << p << '\n';
            // }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#define N_COMM_KIND 5
enum CommKind {
    WORLD,
    NODE,
    CROSS_NODE,
    TREE,
    CROSS
};

struct comm_graph {
    const MpiComms *comms;

    int height;
    int node_size;
    int tree_size;
    int leaf_count;
    int nodes_pre_leaf;

    MyMpiComm inccomm;
    int *inclist, incsize;
    std::function<bool(int,int,int)> incfunc;

    void init(const MpiComms *communicators) {
        comms = communicators;

        height         = communicators->n_tree_comms;
        node_size      = communicators->node_comm.size;
        tree_size      = communicators->tree_comms[0].size;
        leaf_count     = communicators->cross_comms[0].size;
        nodes_pre_leaf = tree_size / node_size;

        incsize = 0;
        inclist = nullptr;
        incfunc = [this](int n, int l, int p) {
            return mywrank(n, l, p);
        };
    }

    bool ininclist(int nodeid, int leafid, int processid) {
        int compare_rank = graphcoo2wrank(nodeid, leafid, processid);
        bool result = false;
        for (int i=0; i<incsize; i++) {
            if (inclist[i] == compare_rank) {
                result = true;
                break;
            }
        }
        return(result);
    }

    void geninclist(CommKind kind, int level=0) {
        if (kind == WORLD) {
            inccomm = comms->world;
        } else if (kind == NODE) {
            inccomm = comms->node_comm;
        } else if (kind == CROSS_NODE) {
            inccomm = comms->cross_comm;
        } else if (kind == TREE){
            inccomm = comms->tree_comms[level];
        } else if (kind == CROSS){
            inccomm = comms->cross_comms[level];
        }

        incsize = inccomm.size;
        inclist = (int*)malloc(sizeof(int)*inccomm.size);
        MPI_Allgather(&(comms->world.rank), 1, MPI_INT, inclist, 1, MPI_INT, inccomm.comm);

        incfunc = [this](int n, int l, int p) {
            return ininclist(n, l, p);
        };

        int name_len = 0;
        char name[MPI_MAX_OBJECT_NAME];
        MPI_Comm_get_name(inccomm.comm, name, &name_len);
        if (comms->world.rank == 0) fprintf(stdout, "Inclusion list updated with %s communicator\n", name);
        MPI_Barrier(comms->world.comm);
    }

    void comms_binary_tree(FILE *fp, int box_w = 7)
    {
        if (height <= 0) {
            fprintf(fp, "(empty tree)\n");
            return;
        }

        int max_nodes = 1 << (height - 1);
        int level_width = max_nodes * (box_w + 2);

        for (int level = 0; level < height; level++) {

            int nodes = 1 << level;
            int spacing = level_width / nodes;

            /* top of boxes */
            for (int i = 0; i < nodes; i++) {
                int pad = spacing - box_w;
                for (int s = 0; s < pad / 2; s++) fprintf(fp, " ");
                fprintf(fp, "*");
                for (int s = 0; s < box_w - 2; s++) fprintf(fp, "-");
                fprintf(fp, "*");
                for (int s = 0; s < pad - pad / 2; s++) fprintf(fp, " ");
            }
            fprintf(fp, "\n");

            /* middle of boxes (fillable area) */
            for (int i = 0; i < nodes; i++) {
                int pad = spacing - box_w;
                for (int s = 0; s < pad / 2; s++) fprintf(fp, " ");
                fprintf(fp, "|");
                for (int s = 0; s < box_w - 2; s++) fprintf(fp, " ");
                fprintf(fp, "|");
                for (int s = 0; s < pad - pad / 2; s++) fprintf(fp, " ");
            }
            fprintf(fp, "\n");

            /* bottom of boxes */
            for (int i = 0; i < nodes; i++) {
                int pad = spacing - box_w;
                for (int s = 0; s < pad / 2; s++) fprintf(fp, " ");
                fprintf(fp, "*");
                for (int s = 0; s < box_w - 2; s++) fprintf(fp, "-");
                fprintf(fp, "*");
                for (int s = 0; s < pad - pad / 2; s++) fprintf(fp, " ");
            }
            fprintf(fp, "\n");

            /* connectors to next level */
            if (level < height - 1) {
                for (int i = 0; i < nodes; i++) {
                    int pad = spacing - box_w;
                    for (int s = 0; s < pad / 2 + box_w / 2 - 1; s++)
                        fprintf(fp, " ");
                    fprintf(fp, "/ \\");
                    for (int s = 0; s < pad - pad / 2 - box_w / 2; s++)
                        fprintf(fp, " ");
                }
                fprintf(fp, "\n");
            }
        }
        return;
    }

    int graphcoo2wrank(int nodeid, int leafid, int processid) {
        return(leafid * tree_size + nodeid * node_size + processid);
    }

    bool mywrank(int nodeid, int leafid, int processid) {
        int compare_rank = graphcoo2wrank(nodeid, leafid, processid);
        return(compare_rank == comms->world.rank);
    }

    void print (FILE *fp) {

        if (inclist != nullptr) {
            int name_len = 0;
            char name[MPI_MAX_OBJECT_NAME];
            MPI_Comm_get_name(inccomm.comm, name, &name_len);
            for (int i=0; i<5*node_size*leaf_count; i++) fprintf(fp, "="); fprintf(fp, "\n");
            fprintf(stdout, "[%d] Current inclusion list was updated to %s communicator\n", comms->world.rank, name);
            for (int i=0; i<5*node_size*leaf_count; i++) fprintf(fp, "="); fprintf(fp, "\n");
        } else {
            for (int i=0; i<5*node_size*leaf_count; i++) fprintf(fp, "="); fprintf(fp, "\n");
            fprintf(stdout, "[%d] No inclusion list selected, incfunc = mywrank\n", comms->world.rank);
            for (int i=0; i<5*node_size*leaf_count; i++) fprintf(fp, "="); fprintf(fp, "\n");
        }

        comms_binary_tree(fp, 3*node_size+4);

        for (int k=0; k<nodes_pre_leaf; k++) {
            for(int j=0; j<leaf_count; j++) {
                fprintf(fp, "  *"); for (int i=0; i<node_size; i++) fprintf(fp, "---"); fprintf(fp, "*  ");
            }
            fprintf(fp, "\n");
            for(int j=0; j<leaf_count; j++) {
                fprintf(fp, "  |"); for (int i=0; i<node_size; i++) fprintf(fp, " _ "); fprintf(fp, "|  ");
            }
            fprintf(fp, "\n");
            for(int j=0; j<leaf_count; j++) {
                fprintf(fp, "  |");
                for (int i=0; i<node_size; i++) {
                    char toprint = (incfunc(k,j,i)) ? 'X' : ' ';
                    fprintf(fp, "|%c|", toprint);
                }
                fprintf(fp, "|  ");
            }
            fprintf(fp, "\n");
            for(int j=0; j<leaf_count; j++) {
                fprintf(fp, "  *"); for (int i=0; i<node_size; i++) fprintf(fp, "---"); fprintf(fp, "*  ");
            }
            fprintf(fp, "\n");
        }
    }
};


/*
 int  assignDeviceToProcess()
{
#ifdef MPI
      char     host_name[MPI_MAX_PROCESSOR_NAME];
      char (*host_names)[MPI_MAX_PROCESSOR_NAME];
      MPI_Comm nodeComm;

#else
      char     host_name[20];
#endif
      int myrank;
      int gpu_per_node;
      int n, namelen, color, rank, nprocs;
      size_t bytes;

#ifdef MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
      MPI_Get_processor_name(host_name,&namelen);

      bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
      host_names = (char (*)[MPI_MAX_PROCESSOR_NAME]) malloc(bytes);

      strcpy(host_names[rank], host_name);

      for (n=0; n<nprocs; n++)
      {
       MPI_Bcast(&(host_names[n]),MPI_MAX_PROCESSOR_NAME, MPI_CHAR,n, MPI_COMM_WORLD);
      }


      qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

      color = 0;

      for (n=0; n<nprocs; n++)
      {
        if(n>0&&strcmp(host_names[n-1], host_names[n])) color++;
        if(strcmp(host_name, host_names[n]) == 0) break;
      }

      MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);

      MPI_Comm_rank(nodeComm, &myrank);
      MPI_Comm_size(nodeComm, &gpu_per_node);
      fprintf(stdout, "DEBUG file %s:%d; rank %d host_names: %s --> myrank: %d, gpu_per_node: %d\n", __FILE__, __LINE__, rank, host_names, myrank, gpu_per_node);

#else
     myrank = 0;
     return 0;
#endif

//      printf ("Assigning device %d  to process on node %s rank %d\n",*myrank,  host_name, rank );
      // Assign device to MPI process, initialize BLAS and probe device properties
      //cudaSetDevice(*myrank);
      return myrank;
}
 */

bool check_node (MpiComms * communicators) {
    char     host_name[MPI_MAX_PROCESSOR_NAME];

    int namelen = 0, reducedlen, reductionflag;
    MPI_Comm nodeComm = communicators->node_comm.comm;
    MPI_Get_processor_name(host_name, &namelen);

    // First check: all ranks in the same node are on the same host
    MPI_Allreduce(&namelen, &reducedlen, 1, MPI_INT, MPI_MIN, nodeComm);
    reductionflag = (reducedlen == namelen) ? 1 : 0 ;
    if (reductionflag == 0) {
        fprintf(stderr, "[%d] hostname mismetch (%d!=%d)\n", communicators->world.rank, reducedlen, namelen);
        fflush(stderr);
    }
    MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN, communicators->world.comm);
    if (reductionflag == 0) return(false);

    char (*host_names)[MPI_MAX_PROCESSOR_NAME];

    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])malloc(communicators->node_comm.size * sizeof(*host_names));

    MPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                nodeComm);

    /* Ensure null-termination */
    host_name[namelen] = '\0';

    for (int i = 0; i < communicators->node_comm.size; i++) {
        if (strcmp(host_names[0], host_names[i]) != 0) {
            fprintf(stderr,
                    "[world %d | node %d] hostname mismatch: '%s' vs '%s'\n",
                    communicators->world.rank,
                    communicators->node_comm.rank,
                    host_names[0], host_names[i]);
            reductionflag = 0;
            break;
        }
    }


    /* Reduce result across world */
    MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN,
                  communicators->world.comm);

    if (!reductionflag) return(false);

    free(host_names);

    /* Second check: ranks on different nodes have different hostnames */
    int cross_size = communicators->cross_comm.size;
    int cross_rank = communicators->cross_comm.rank;

    host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])
        malloc(cross_size * sizeof(*host_names));

    /* Gather one hostname per node */
    MPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                communicators->cross_comm.comm);

    /* Compare all hostnames: must be unique */
    for (int i = 0; i < cross_size; i++) {
        for (int j = i + 1; j < cross_size; j++) {
            if (strcmp(host_names[i], host_names[j]) == 0) {
                fprintf(stderr,
                        "[world %d | cross %d] node hostname collision: '%s'\n",
                        communicators->world.rank,
                        cross_rank,
                        host_names[i]);
                reductionflag = 0;
                break;
            }
        }
        if (!reductionflag) break;
    }

    /* Ensure all ranks agree */
    MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN,
                communicators->world.comm);

    free(host_names);

    if (!reductionflag) return(false);


    return(true);
}

#endif

