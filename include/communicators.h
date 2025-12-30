#pragma once

#include "common.h"
#include <functional>
#include <string>
#include <vector>
#include <iostream>

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

    void init_processenv(void) {
        home = std::getenv("HOME");
        slurm_node = std::getenv("SLURM_NODEID");
        slurm_addr = std::getenv("SLURM_TOPOLOGY_ADDR");
    }

} ProcessEnv;

typedef struct mpi_comms
{
    MyMpiComm world;
    MyMpiComm node_comm;
    MyMpiComm cross_comm;

    int n_tree_comms;
    MyMpiComm *tree_comms;
    MyMpiComm *root_comms;
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
    communicators->tree_comms   = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(communicators->n_tree_comms)); // check > 0
    communicators->cross_comms  = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(communicators->n_tree_comms)); // check > 0
    communicators->root_comms   = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(communicators->n_tree_comms)); // check > 0
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

        // int root_comm_id = communicators->cross_comm.rank / (config->npl * subtree_count);
        MyMpiComm splitcomm = (i==0) ? (communicators->cross_comm)  : (communicators->cross_comms[0]) ;
        int root_comm_id    = (i==0) ? (splitcomm.rank/config->npl) : (splitcomm.rank / subtree_count);
        MPI_Comm_split(splitcomm.comm, root_comm_id, wrank, &tmp_comm);
        init_mympicomm(tmp_comm, &(communicators->root_comms[i]));

        snprintf(comm_name, 50, "RootLevel%dComm", i);
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

            for (int i = 0; i < communicators->n_tree_comms; i++) {
                char label[32];
                snprintf(label, sizeof(label), "ROOT[%d]", i);
                print_comm_info(label, &communicators->root_comms[i], fp);
            }

            fprintf(fp, "[%-12s] HOME=%-20s\n",
                   "ENV-VARS", communicators->myenv.home);

            fprintf(fp, "[%-12s] SLURM_NODEID=%-5s, SLURM_TOPOLOGY_ADDR=%-20s\n",
                "SLURM-ENV", communicators->myenv.slurm_node, communicators->myenv.slurm_addr);
            fflush(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

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
        } else if (kind == CROSS_TREE){
            inccomm = comms->cross_comms[level];
        } else if (kind == ROOT){
            inccomm = comms->root_comms[level];
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

int addr_distance(const char *a, const char *b) {
    int i = 0;
    int last_dot = -1;
    int dots = 0;

    while (a[i] && b[i] && a[i] == b[i]) {
        if (a[i] == '.') {
            dots++;
            last_dot = i;
        }
        i++;
    }

    // If mismatch occurred after a dot, dots is correct.
    // If mismatch occurred inside a field, ignore the partial field.
    // dots already counts only completed fields.

    /* if both reached NUL, strings are identical -> add +1 */
    if (a[i] == '\0' && b[i] == '\0') dots++;
    return dots;
}

char** agg_addrstrings (MpiComms * communicators, CommKind type, int level = 0) {

    MyMpiComm mycomm;
    if (type == WORLD) {
        mycomm = communicators->world;
    } else if (type == NODE) {
        mycomm = communicators->node_comm;
    } else if (type == CROSS_NODE) {
        mycomm = communicators->cross_comm;
    } else if (type == TREE){
        mycomm = communicators->tree_comms[level];
    } else if (type == CROSS_TREE){
        mycomm = communicators->cross_comms[level];
    } else if (type == ROOT){
        mycomm = communicators->root_comms[level];
    }

    int *alllen = (int*)malloc(sizeof(int)*mycomm.size);
    int  mylen  = (int)strlen(communicators->myenv.slurm_addr);
    MPI_Allgather(&mylen, 1, MPI_INT, alllen, 1, MPI_INT, mycomm.comm);

    /* compute prefix-sum (displacements) and total bytes */
    int *displs = (int*)malloc(sizeof(int)*mycomm.size);
    if (!displs) {
        free(alllen);
        return nullptr;
    }

    displs[0] = 0;
    for (int i = 1; i < mycomm.size; ++i) displs[i] = displs[i-1] + alllen[i-1];
    int total_bytes = displs[mycomm.size-1] + alllen[mycomm.size-1];

    /* allocate receive buffer for concatenated addresses */
    char *allbuf = nullptr;
    if (total_bytes > 0) {
        allbuf = (char*)malloc(sizeof(char)*total_bytes);
        if (!allbuf) {
            free(alllen);
            free(displs);
            return nullptr;
        }
    }

    // Receve the allbuffs
    MPI_Allgatherv(communicators->myenv.slurm_addr, mylen, MPI_CHAR, allbuf, alllen, displs, MPI_CHAR, mycomm.comm);

    /* Reconstruct an array of NUL-terminated strings from the concatenated buffer */
    char **all_addrs = (char**)malloc(sizeof(char*)*mycomm.size);
    if (!all_addrs) {
        free(allbuf);
        free(alllen);
        free(displs);
        return nullptr;
    }

    for (int i = 0; i < mycomm.size; ++i) {
        int len = alllen[i];
        if (len > 0) {
            char *s = (char*) malloc((size_t)len + 1);
            if (!s) {
                /* cleanup partial allocations */
                for (int j = 0; j < i; ++j) free(all_addrs[j]);
                free(all_addrs); free(allbuf); free(alllen); free(displs);
                return nullptr;
            }
            memcpy(s, allbuf + displs[i], (size_t)len);
            s[len] = '\0';
            all_addrs[i] = s;
        } else {
            /* empty string */
            all_addrs[i] = (char*) malloc(1);
            if (!all_addrs[i]) {
                for (int j = 0; j < i; ++j) free(all_addrs[j]);
                free(all_addrs); free(allbuf); free(alllen); free(displs);
                return nullptr;
            }
            all_addrs[i][0] = '\0';
        }
    }

    free(allbuf);
    free(alllen);
    free(displs);

    return(all_addrs);
}

bool check_addr (MpiComms * communicators) {

    bool check_flag  = true;
    int wrank = communicators->world.rank;
    const char *myaddr = communicators->myenv.slurm_addr;

    if (myaddr == nullptr) {
        if (wrank == 0)
            fprintf(stdout, "ADDR check failed, addr var was not populated\n");
        return(false);
    }

    // Check node level
    MyMpiComm currentcomm = communicators->node_comm;
    char **node_addr = agg_addrstrings(communicators, NODE);
    for (int i=0; i<currentcomm.size; i++) {
        if (i != currentcomm.rank) {
            int addr_dist = addr_distance(myaddr, node_addr[i]);
            if (addr_dist != communicators->n_tree_comms + 1) {
                int  name_len = 0;
                char name[MPI_MAX_OBJECT_NAME];
                MPI_Comm_get_name(currentcomm.comm, name, &name_len);
                fprintf(stderr, "[%d] Error: dist with %d on comm %s\n", wrank, i, name);
                check_flag = false;
                break;
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &check_flag, 1, MPI_C_BOOL, MPI_LAND, currentcomm.comm);


    if (wrank == 0) {
        fprintf(stdout, "---------- Check node level ----------\n");
        for (int i=0; i<currentcomm.size; i++) {
            int addr_dist = addr_distance(myaddr, node_addr[i]);
            fprintf(stdout, "[0 from %d] addr_string: %s, addr_dist: %d\n", i, node_addr[i], addr_dist);
        }

        fprintf(stdout, "addr_dist: ");
        for (int i=0; i<currentcomm.size; i++) {
            for (int j=i+1; j<currentcomm.size; j++) {
                int addr_dist = addr_distance(node_addr[i], node_addr[j]);
                fprintf(stdout, "(%d,%d)=%d ", i, j, addr_dist);
            }
        }
        fprintf(stdout, "\n");
    }
    for (int i = 0; i < currentcomm.size; i++) free(node_addr[i]); free(node_addr);
    MPI_Barrier(communicators->world.comm);


    // Check tree levels
    char **tree_addr;
    for (int level=0; level<communicators->n_tree_comms; level++) {

        currentcomm = communicators->root_comms[level];
        tree_addr = agg_addrstrings(communicators, ROOT, level);
        for (int i=0; i<currentcomm.size; i++) {
            if (i != currentcomm.rank) {
                int addr_dist = addr_distance(myaddr, tree_addr[i]);
                if (addr_dist != communicators->n_tree_comms - level) {
                    int  name_len = 0;
                    char name[MPI_MAX_OBJECT_NAME];
                    MPI_Comm_get_name(currentcomm.comm, name, &name_len);
                    fprintf(stderr, "[%d] Error: dist with %d on comm %s\n", wrank, i, name);
                    check_flag = false;
                    break;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &check_flag, 1, MPI_C_BOOL, MPI_LAND, currentcomm.comm);


        if (wrank == 0) {
            fprintf(stdout, "---------- Check tree level %d level ----------\n", level);
            for (int i=0; i<currentcomm.size; i++) {
                int addr_dist = addr_distance(myaddr, tree_addr[i]);
                fprintf(stdout, "[0 from %d] addr_string: %s, addr_dist: %d\n", i, tree_addr[i], addr_dist);
            }

            fprintf(stdout, "addr_dist: ");
            for (int i=0; i<currentcomm.size; i++) {
                for (int j=i+1; j<currentcomm.size; j++) {
                    int addr_dist = addr_distance(tree_addr[i], tree_addr[j]);
                    fprintf(stdout, "(%d,%d)=%d ", i, j, addr_dist);
                }
            }
            fprintf(stdout, "\n");
        }
        for (int i = 0; i < currentcomm.size; i++) free(tree_addr[i]); free(tree_addr);
        MPI_Barrier(communicators->world.comm);
    }

    return(check_flag);
}
