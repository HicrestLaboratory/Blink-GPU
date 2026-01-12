#pragma once

#include "common.h"
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include "cmd_util.h"

struct MpiComms
{
    MyMpiComm world;
    MyMpiComm node_comm;
    MyMpiComm cross_comm;

    int n_tree_comms;
    MyMpiComm *tree_comms;
    MyMpiComm *root_comms;
    MyMpiComm *cross_comms;

    ProcessEnv myenv;

    void init(Config * config) {

        world.init(MPI_COMM_WORLD);

        // Check correct process num
        if (world.size != config->nprocess) {
            fprintf(stderr, "Error: mismetch between defined processes (%d) and MPI processes (%d)\n", config->nprocess, world.size);
            exit(__LINE__);
        }

        MPI_Comm tmp_comm;
        int wrank    = world.rank;
        int node_id  = wrank / (config->ppn);
        int cross_id = wrank % (config->ppn);

        MPI_Comm_split(MPI_COMM_WORLD, node_id,  wrank, &tmp_comm);
        node_comm.init(tmp_comm);
        MPI_Comm_set_name(tmp_comm, "NodeComm");

        MPI_Comm_split(MPI_COMM_WORLD, cross_id, wrank, &tmp_comm);
        cross_comm.init(tmp_comm);
        MPI_Comm_set_name(tmp_comm, "CrossNodeComm");

        int process_per_leaf = (config->ppn * config->npl);
        n_tree_comms = config->tree_high;
        tree_comms   = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(n_tree_comms)); // check > 0
        cross_comms  = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(n_tree_comms)); // check > 0
        root_comms   = (MyMpiComm*)malloc(sizeof(MyMpiComm)*(n_tree_comms)); // check > 0
        for (int i=0; i<n_tree_comms; i++) {
            int subtree_count   = 1<<i;

            int tree_comm_size  = process_per_leaf * subtree_count;
            int tree_comm_id    = wrank / tree_comm_size;
            MPI_Comm_split(MPI_COMM_WORLD, tree_comm_id, wrank, &tmp_comm);
            tree_comms[i].init(tmp_comm);

            char comm_name[50];
            snprintf(comm_name, 50, "Level%dTreeComm", i);
            MPI_Comm_set_name(tmp_comm, comm_name);

            int cross_comm_id    = wrank % tree_comm_size;
            MPI_Comm_split(MPI_COMM_WORLD, cross_comm_id, wrank, &tmp_comm);
            cross_comms[i].init(tmp_comm);

            snprintf(comm_name, 50, "CrossLevel%dComm", i);
            MPI_Comm_set_name(tmp_comm, comm_name);

            MyMpiComm splitcomm = (i==0) ? (cross_comm)  : (cross_comms[0]) ;
            int root_comm_id    = (i==0) ? (splitcomm.rank/config->npl) : (splitcomm.rank / subtree_count);
            MPI_Comm_split(splitcomm.comm, root_comm_id, wrank, &tmp_comm);
            root_comms[i].init(tmp_comm);

            snprintf(comm_name, 50, "RootLevel%dComm", i);
            MPI_Comm_set_name(tmp_comm, comm_name);
        }

        myenv.init_processenv();
    }

    void info(FILE *fp=stdout) {
        int world_rank = world.rank;

        if (world_rank == 0) {
            int len;
            char version[MPI_MAX_LIBRARY_VERSION_STRING];
            MPI_Get_library_version(version, &len);
            printf("%s\n", version);
        }

        /* Optional: make output readable by rank */
        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < world.size; r++) {
            if (r == world_rank) {
                fprintf(fp, "\n=== Communicator info (world rank %d) ===\n", r);

                print_comm_info("WORLD",      &world, fp);
                print_comm_info("NODE",       &node_comm, fp);
                print_comm_info("CROSS-NODE", &cross_comm, fp);

                for (int i = 0; i < n_tree_comms; i++) {
                    char label[32];
                    snprintf(label, sizeof(label), "TREE[%d]", i);
                    print_comm_info(label, &tree_comms[i], fp);
                }

                for (int i = 0; i < n_tree_comms; i++) {
                    char label[32];
                    snprintf(label, sizeof(label), "CROSS[%d]", i);
                    print_comm_info(label, &cross_comms[i], fp);
                }

                for (int i = 0; i < n_tree_comms; i++) {
                    char label[32];
                    snprintf(label, sizeof(label), "ROOT[%d]", i);
                    print_comm_info(label, &root_comms[i], fp);
                }

                fprintf(fp, "[%-12s] HOME=%-20s\n",
                    "ENV-VARS", myenv.home);

                fprintf(fp, "[%-12s] SLURM_NODEID=%-5s, SLURM_TOPOLOGY_ADDR=%-20s\n",
                    "SLURM-ENV", myenv.slurm_node, myenv.slurm_addr);
                fflush(fp);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    bool check_node (void) {
        char     host_name[MPI_MAX_PROCESSOR_NAME];

        int namelen = 0, reducedlen, reductionflag;
        MPI_Comm nodeComm = node_comm.comm;
        MPI_Get_processor_name(host_name, &namelen);

        // First check: all ranks in the same node are on the same host
        MPI_Allreduce(&namelen, &reducedlen, 1, MPI_INT, MPI_MIN, nodeComm);
        reductionflag = (reducedlen == namelen) ? 1 : 0 ;
        if (reductionflag == 0) {
            fprintf(stderr, "[%d] hostname mismetch (%d!=%d)\n", world.rank, reducedlen, namelen);
            fflush(stderr);
        }
        MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN, world.comm);
        if (reductionflag == 0) return(false);

        char (*host_names)[MPI_MAX_PROCESSOR_NAME];

        host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])malloc(node_comm.size * sizeof(*host_names));

        MPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                    host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                    nodeComm);

        /* Ensure null-termination */
        host_name[namelen] = '\0';

        for (int i = 0; i < node_comm.size; i++) {
            if (strcmp(host_names[0], host_names[i]) != 0) {
                fprintf(stderr,
                        "[world %d | node %d] hostname mismatch: '%s' vs '%s'\n",
                        world.rank,
                        node_comm.rank,
                        host_names[0], host_names[i]);
                reductionflag = 0;
                break;
            }
        }


        /* Reduce result across world */
        MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN, world.comm);

        if (!reductionflag) return(false);

        free(host_names);

        /* Second check: ranks on different nodes have different hostnames */
        int cross_size = cross_comm.size;
        int cross_rank = cross_comm.rank;

        host_names = (char (*)[MPI_MAX_PROCESSOR_NAME])
            malloc(cross_size * sizeof(*host_names));

        /* Gather one hostname per node */
        MPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                    host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                    cross_comm.comm);

        /* Compare all hostnames: must be unique */
        for (int i = 0; i < cross_size; i++) {
            for (int j = i + 1; j < cross_size; j++) {
                if (strcmp(host_names[i], host_names[j]) == 0) {
                    fprintf(stderr,
                            "[world %d | cross %d] node hostname collision: '%s'\n",
                            world.rank,
                            cross_rank,
                            host_names[i]);
                    reductionflag = 0;
                    break;
                }
            }
            if (!reductionflag) break;
        }

        /* Ensure all ranks agree */
        MPI_Allreduce(MPI_IN_PLACE, &reductionflag, 1, MPI_INT, MPI_MIN, world.comm);

        free(host_names);

        if (!reductionflag) return(false);


        return(true);
    }

    char** agg_addrstrings (CommKind type, int level = 0) {

        MyMpiComm mycomm;
        if (type == WORLD) {
            mycomm = world;
        } else if (type == NODE) {
            mycomm = node_comm;
        } else if (type == CROSS_NODE) {
            mycomm = cross_comm;
        } else if (type == TREE){
            mycomm = tree_comms[level];
        } else if (type == CROSS_TREE){
            mycomm = cross_comms[level];
        } else if (type == ROOT){
            mycomm = root_comms[level];
        }

        int *alllen = (int*)malloc(sizeof(int)*mycomm.size);
        int  mylen  = (int)strlen(myenv.slurm_addr);
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
        MPI_Allgatherv(myenv.slurm_addr, mylen, MPI_CHAR, allbuf, alllen, displs, MPI_CHAR, mycomm.comm);

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

    bool check_addr(void) {

        bool check_flag  = true;
        int wrank = world.rank;
        const char *myaddr = myenv.slurm_addr;

        if (myaddr == nullptr) {
            if (wrank == 0)
                fprintf(stdout, "ADDR check failed, addr var was not populated\n");
            return(false);
        }

        // Check node level
        MyMpiComm currentcomm = node_comm;
        char **node_addr = agg_addrstrings(NODE);
        for (int i=0; i<currentcomm.size; i++) {
            if (i != currentcomm.rank) {
                int addr_dist = addr_distance(myaddr, node_addr[i]);
                if (addr_dist != n_tree_comms + 1) {
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
        MPI_Barrier(world.comm);


        // Check tree levels
        char **tree_addr;
        for (int level=0; level<n_tree_comms; level++) {

            currentcomm = root_comms[level];
            tree_addr = agg_addrstrings(ROOT, level);
            for (int i=0; i<currentcomm.size; i++) {
                if (i != currentcomm.rank) {
                    int addr_dist = addr_distance(myaddr, tree_addr[i]);
                    if (addr_dist != n_tree_comms - level) {
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
            MPI_Barrier(world.comm);
        }

        return(check_flag);
    }

    void assign_cuda_gpu(void) {
        int num_devices = 0;
        cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
        MPI_Allreduce(MPI_IN_PLACE, &num_devices, 1, MPI_INT, MPI_MIN, cross_comm.comm);

        if (num_devices != node_comm.size) {
            fprintf(stderr, "Error: ngpus per node must be the same on all the nodes and must be the same of the nodeComm size.\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
        cudaSetDevice(node_comm.rank);
    }
};

struct BlinkCommWrapper {
    MpiComms *comms;

    int height;
    int node_size;
    int tree_size;
    int leaf_count;
    int nodes_pre_leaf;

    MyMpiComm inccomm;
    int *inclist, incsize;
    std::function<bool(int,int,int)> incfunc;

    void init(MpiComms *communicators) {
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

    void init(Config *config, bool gpu_flag, CommKind kind=WORLD, int level=0) {
        comms = (MpiComms*)malloc(sizeof(MpiComms));
        comms->init(config);

        bool test = comms->check_node();
        if (comms->world.rank == 0) fprintf(stdout, "Node check %s\n", (test) ? "true" : "false");

        if (gpu_flag) comms->assign_cuda_gpu();

        height         = comms->n_tree_comms;
        node_size      = comms->node_comm.size;
        tree_size      = comms->tree_comms[0].size;
        leaf_count     = comms->cross_comms[0].size;
        nodes_pre_leaf = tree_size / node_size;

        incsize = 0;
        inclist = nullptr;
        incfunc = [this](int n, int l, int p) {
            return mywrank(n, l, p);
        };

        geninclist(kind, level);
    }

#ifdef NCCL
    void add_nccl(void) {
        inccomm.add_nccl();
    }
#endif

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
