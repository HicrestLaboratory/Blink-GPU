#include "cmd_util.h"
#include "communicators.h"

struct AddrStruct {
    const char *addr_str;
    int         addr_str_len;

    char *node_name;
    int   node_name_len;

    int    nswitchs;
    char **switch_names;
    int   *switch_name_lens;

    int max_lable_len;

    void init(const char *addr_string)
    {
        addr_str     = addr_string;
        addr_str_len = strlen(addr_string);

        int dots = 0;
        for (int i = 0; addr_string[i]; i++) {
            if (addr_string[i] == '.') dots++;
        }

        /* tokens = dots + 1 */
        nswitchs = dots;   /* last token is node_name */

        switch_names     = (char **)malloc(sizeof(char *) * nswitchs);
        switch_name_lens = (int   *)malloc(sizeof(int)    * nswitchs);

        const char *start = addr_string;
        const char *dot   = NULL;

        /* Extract switch names */
        for (int j = 0; j < nswitchs; j++) {
            dot = strchr(start, '.');

            switch_name_lens[j] = dot - start;
            switch_names[j] = (char *)malloc(switch_name_lens[j] + 1);

            memcpy(switch_names[j], start, switch_name_lens[j]);
            switch_names[j][switch_name_lens[j]] = '\0';

            start = dot + 1;
        }

        /* Remaining token → node_name */
        node_name_len = strlen(start);
        node_name = (char *)malloc(node_name_len + 1);
        memcpy(node_name, start, node_name_len);
        node_name[node_name_len] = '\0';

        // Compute local max name len
        max_lable_len = strlen(node_name);
        for (int i=0; i<nswitchs; i++) if(strlen(switch_names[i])>max_lable_len) max_lable_len = strlen(switch_names[i]);
    }

    void print(FILE *fp) {
        fprintf(fp, "\n=== ADDR info  ===\n");
        fprintf(fp, "ADDR string: %s\n\n", addr_str);

        for(int i=0; i<nswitchs; i++) fprintf(fp, "Switch L%d: %s\n", nswitchs - i, switch_names[i]);
        fprintf(fp, "Node: %s\n", node_name);
    }
};

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

char **allgather_strings(int strlen_local, const char *str, MPI_Comm mycomm)
{
    int size;
    MPI_Comm_size(mycomm, &size);

    /* 1. Gather string lengths (WITHOUT '\0') */
    int *lengths = (int *)malloc(sizeof(int) * size);
    MPI_Allgather(&strlen_local, 1, MPI_INT,
                  lengths,         1, MPI_INT,
                  mycomm);

    /* 2. Compute displacements */
    int *displs = (int *)malloc(sizeof(int) * size);
    int total_len = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = total_len;
        total_len += lengths[i];
    }

    /* 3. Gather raw bytes */
    char *aggbuf = (char *)malloc(sizeof(char) * total_len);
    MPI_Allgatherv(str, strlen_local, MPI_CHAR,
                   aggbuf, lengths, displs, MPI_CHAR,
                   mycomm);

    /* 4. Build char** vector */
    char **strings = (char **)malloc(sizeof(char *) * size);

    for (int i = 0; i < size; i++) {
        strings[i] = (char *)malloc(lengths[i] + 1);
        memcpy(strings[i], aggbuf + displs[i], lengths[i]);
        strings[i][lengths[i]] = '\0';
    }

    /* 5. Cleanup temporaries */
    free(lengths);
    free(displs);
    free(aggbuf);

    return strings;
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

std::vector<char*> gen_ordering_vec(char **agg, int len) {
    std::vector<char*> ordering_vec;

    int j;
    for (int i = 0; i < len; i++) {
        for (j = 0; j < ordering_vec.size(); j++)
            if (strcmp(agg[i], ordering_vec[j]) == 0) break;

        if (j==ordering_vec.size()) ordering_vec.push_back(agg[i]);
    }

    return(ordering_vec);
}

int myorder(const std::vector<char*>& v,
          const char* elem0,
          const char* elem1)
{
    if (!elem0 || !elem1)
        return -1;

    for (size_t i = 0; i < v.size(); ++i) {
        const char* s = v[i];
        if (!s) continue;

        if (strcmp(s, elem0) == 0)
            return 1;

        if (strcmp(s, elem1) == 0)
            return 0;
    }

    return -1;
}

int find(const std::vector<char*>& v, char* elem) {
    for (size_t i = 0; i < v.size(); i++)
        if (strcmp(v[i], elem) == 0) return(i);

    return(-1);
}

struct ComputeNewWorld {
    AddrStruct myAddrInfo;
    MPI_Comm    old_world;
    MPI_Comm    new_world;

    int     nfields;
    char  **agg_str;
    char ***agg_vec;

    int     *subcomm_colors;
    MPI_Comm      *subcomms;
    MPI_Comm    *crosscomms;
    MPI_Comm *subcrosscomms;

    std::vector<std::vector<char*>> ord_vec;

    int *new_ranks;
    int  size, old_rank, new_rank;

    void init(const AddrStruct& addr_input, const MPI_Comm& input_comm) {
        old_world = input_comm;
        myAddrInfo = addr_input;
        MPI_Comm_size(old_world, &size);
        MPI_Comm_rank(old_world, &old_rank);

        // Aggregate raw ADDR string
        agg_str = allgather_strings(myAddrInfo.addr_str_len, myAddrInfo.addr_str, old_world);
        // if (old_rank == 0) print_array_inline(agg_str, size, "agg_str");

        // Aggregate field by field ADDR info to obtain vectors
        nfields = myAddrInfo.nswitchs + 1;
        agg_vec = (char***)malloc(sizeof(char**)*nfields);
        for (int j=0; j<nfields; j++) {
            int    len2agg = (j<nfields-1) ? myAddrInfo.switch_name_lens[j] : myAddrInfo.node_name_len ;
            char *char2agg = (j<nfields-1) ? myAddrInfo.switch_names[j]     : myAddrInfo.node_name     ;
            agg_vec[j] = allgather_strings(len2agg, char2agg, old_world);
            // if (old_rank == 0) print_array_inline(agg_vec[j], size, "agg_vec");
        }

        // Obtain ordering vectors
        for (int j=0; j<nfields; j++) {
            std::vector<char*> tmp = gen_ordering_vec(agg_vec[j], size);
            ord_vec.push_back(tmp);
            // if(old_rank == 0) print_vector_inline(tmp, "ord_vec");
        }
    }

    MPI_Comm gen_new_world(void) {

        // Alloc and init new_ranks
        new_ranks = (int*)malloc(sizeof(int)*size);
        for (int i=0; i<size; i++) new_ranks[i] = 0;
        // if (old_rank == 5) print_array_inline(new_ranks, size, "new_ranks");

        // Generate local new_ranks contributions
        for (int i=0; i<size; i++) {
            int level = addr_distance(myAddrInfo.addr_str, agg_str[i]);
            // if (old_rank == 5) fprintf(stdout, "addr_distance at %d: %d (%s, %s)\n", i, level, myAddrInfo.addr_str, agg_str[i]);

            if (level == nfields) {
                new_ranks[i] = (old_rank < i) ? 1 : 0; // We are in the same node
            } else {
                char *him  = agg_vec[level][i];
                char *mine = (level == myAddrInfo.nswitchs) ? myAddrInfo.node_name : myAddrInfo.switch_names[level] ;
                const std::vector<char*>& order_vec = ord_vec[level];

                new_ranks[i] = myorder(order_vec, mine, him);
            }
        }
        // if (old_rank == 5) print_array_inline(new_ranks, size, "new_ranks");

        // Aggregate new_ranks contributions
        MPI_Allreduce(MPI_IN_PLACE, new_ranks, size, MPI_INT, MPI_SUM, old_world);
        // if (old_rank == 5) print_array_inline(new_ranks, size, "new_ranks");

        // Generate new_world communicator
        new_rank = new_ranks[old_rank];
        MPI_Comm_split(old_world, 0, new_rank, &new_world);

        return(new_world);
    }

    MPI_Comm* gen_sub_comms(void) {

        // Compute colors to split
        subcomm_colors = (int*)malloc(sizeof(int)*nfields);
        for (int level=0; level<nfields; level++) {
            char *to_search = (level == myAddrInfo.nswitchs) ? myAddrInfo.node_name : myAddrInfo.switch_names[level] ;
            std::vector<char*> order_vec = ord_vec[level];

            subcomm_colors[level] = find(order_vec, to_search);
        }

        // Generate sub communicators
        subcomms = (MPI_Comm*)malloc(sizeof(MPI_Comm)*nfields);
        for (int level=0; level<nfields; level++) {
            MPI_Comm_split(new_world, subcomm_colors[level], new_rank, &subcomms[level]);
        }

        // free(subcomm_colors);
        return(subcomms);
    }

    MPI_Comm* gen_cross_comms(void) {

        // Generate sub communicators
        crosscomms    = (MPI_Comm*)malloc(sizeof(MPI_Comm)*nfields);
        subcrosscomms = (MPI_Comm*)malloc(sizeof(MPI_Comm)*nfields);
        for (int level=0; level<nfields; level++) {

            int subcomm_size, subcomm_rank;
            MPI_Comm_size(subcomms[level], &subcomm_size);
            MPI_Comm_rank(subcomms[level], &subcomm_rank);
            int color = (subcomm_rank == 0) ? 0 : MPI_UNDEFINED ;

            MPI_Comm startcomm = (level > 0) ? (subcomms[level-1]) : new_world ;
            MPI_Comm_split(startcomm, color, new_rank, &subcrosscomms[level]);
            MPI_Comm_split(new_world, color, new_rank, &crosscomms[level]);
        }
        return(crosscomms);
    }

    void clear(void) {
        // Free agg_str
        if (agg_str) {
            for (int i = 0; i < size; ++i) {
                free(agg_str[i]);   // individual strings
            }
            free(agg_str);
            agg_str = nullptr;
        }

        // Free agg_vec
        if (agg_vec) {
            for (int j = 0; j < nfields; ++j) {
                if (agg_vec[j]) {
                    for (int i = 0; i < size; ++i) {
                        free(agg_vec[j][i]);  // individual strings
                    }
                    free(agg_vec[j]);
                }
            }
            free(agg_vec);
            agg_vec = nullptr;
        }

        // Free new_ranks
        if (new_ranks) {
            free(new_ranks);
            new_ranks = nullptr;
        }

        // Clear vectors (non-owning)
        ord_vec.clear();

        // Reset scalars (defensive)
        nfields  = 0;
        size     = 0;
        old_rank = -1;
        new_rank = -1;
    }


    void gen_all(void) {
        gen_new_world();
        gen_sub_comms();
        gen_cross_comms();
    }

};

const char* emptyStr(int nrep) {
    static char s[20000];
    for (int i=0; i<nrep; i++) sprintf(s + i, "%s", " ");
    return(s);
}

const char* patternStr(const char* pad, int nrep) {
    static char s[20000];
    for (int i=0; i<nrep; i++) sprintf(s + i*strlen(pad), "%s", pad);
    return(s);
}

const char* printLable(const char* lable, int size) {
    static char s[20000];
    int npad = size - strlen(lable);
    int sshift = (npad/2 + npad%2);
    int fshift = (npad/2);
    for (int i=0; i<sshift; i++) sprintf(s + i, " ");
    sprintf(s + sshift, "%s", lable);
    for (int i=0; i<fshift; i++) sprintf(s + sshift + strlen(lable) + i, " ");
    return(s);
}

struct NetworkGraph {
    int nl1;     // Number of L1 switches (i.e. lower level switches)
    int shight;  // High of the switch tree
    int max_nps; // Max number of nodes per L1 switch
    int max_ppn; // Max number of processes per node

    // Widths for the printing
    int l1_width;
    int node_width;
    int max_lable_len;

    // ---------- Graphs ----------
    /* Node graph:
     * nl1 vectors, each one with the current format:
     *     gnodes[i] = {"Nj0_name", processes in Nj0}, {"Nj1_name", processes in Nj1}, ...
     *
     * Where Nj0 is the first node under the i-th L1 switch, Nj1 the second and so on
     */
    std::vector<std::pair<char*,int>> *gnodes;

    /* Switch graph:
     * shight vectors, each one with the current format:
     *     gnodes[i] = {"S0_name", #L1 switches under S0}, {"S1_name", #L1 switches under S1}, ...
     *
     * Where S0 is the first i-th level switch in the network, S1 the second and so on. Each level has number of elements between 1 and nl1.
     * L1 switches are all the L1 switches under the switch (like the number of leafs under a tree vertex). Last vector represent the L1 switches,
     * meaning that all the sizes are equal to 1.
     */
    std::vector<std::pair<char*,int>> *gswitch;

    void shortPrint(FILE *fp = stdout) {
        std::cout << " --------------------------------------------------------- " << std::endl;
        std::cout << "                       Network Graph                       " << std::endl;
        std::cout << " --------------------------------------------------------- " << std::endl;

        for(int level=0; level<shight; level++) {
            std::cout << "graph_switch[" << level << "]: ";
            for (std::pair<char*,int> e: gswitch[level]) std::cout << '{' << std::get<0>(e) << ',' << std::get<1>(e) << "}," ;
            std::cout << std::endl;
        }

        for(int level=0; level<nl1; level++) {
            std::cout << "graph_nodes[" << level << "]: ";
            for (std::pair<char*,int> e: gnodes[level]) std::cout << '{' << std::get<0>(e) << ',' << std::get<1>(e) << "}," ;
            std::cout << std::endl;
        }
    }

    void netPrint(FILE *fp = stdout) {
        std::cout << " --------------------------------------------------------- " << std::endl;
        std::cout << "                       Network Graph                       " << std::endl;
        std::cout << " --------------------------------------------------------- " << std::endl;

        for(int level=0; level<shight*3; level++) {
            for (std::pair<char*,int> e: gswitch[level/3]) {
                if (level%3 != 1) fprintf(fp, "*%s*", patternStr("-", std::get<1>(e)*l1_width + 2*(std::get<1>(e)-1)));
                else fprintf(fp, "|%s|", printLable(std::get<0>(e), std::get<1>(e)*l1_width + 2*(std::get<1>(e)-1)));
            }
            fprintf(fp, "\n");
        }

        for (int j=0; j<max_nps*5; j++) {
            for (int i=0; i<nl1; i++) {
                auto e = gnodes[i];
                // fprintf(fp, "(%d,%d) %zu. %d\n", j, i, e.size(), e.size() > j/3);
                if (e.size() > j/5) {
                    if (j%5 == 1) fprintf(fp, " |%s| ", printLable(std::get<0>(e[j/5]), node_width));
                    else if (j%5 == 2) {
                        fprintf(fp, " |%s%s| ", patternStr(" _ ", std::get<1>(e[j/5])), emptyStr(node_width-3*std::get<1>(e[j/5])));
                    } else if (j%5 == 3) {
                        fprintf(fp, " |%s%s| ", patternStr("| |", std::get<1>(e[j/5])), emptyStr(node_width-3*std::get<1>(e[j/5])));
                    } else fprintf(fp, " *%s* ", patternStr("-", node_width));
                } else {
                    fprintf(fp, "  %s  ", patternStr(" ", node_width));
                }
            }
            fprintf(fp, "\n");
        }
    }
};

struct MpiComms2 {
    MyMpiComm world;
    AddrStruct addr;

    int              nfields;
    MyMpiComm      *subcomms;
    MyMpiComm    *crosscomms;
    MyMpiComm *subcrosscomms;

    NetworkGraph graph;

    void init(const ComputeNewWorld& input) {
        nfields = input.nfields;
        world.init(input.new_world);
        addr = input.myAddrInfo;

        subcomms      = (MyMpiComm*)malloc(sizeof(MyMpiComm)*nfields);
        crosscomms    = (MyMpiComm*)malloc(sizeof(MyMpiComm)*nfields);
        subcrosscomms = (MyMpiComm*)malloc(sizeof(MyMpiComm)*nfields);
        for (int i=0; i<nfields; i++) {
            subcomms[i].init(input.subcomms[i]);
            crosscomms[i].init(input.crosscomms[i]);
            subcrosscomms[i].init(input.subcrosscomms[i]);
        }
    }

    void pregraph (void) {
        int max_nps, nps = (subcrosscomms[nfields-1].comm != MPI_COMM_NULL) ? subcrosscomms[nfields-1].size : 0;
        MPI_Bcast(&nps, 1, MPI_INT, 0, subcomms[nfields-1].comm);
        MPI_Allreduce(&nps, &max_nps, 1, MPI_INT, MPI_MAX, world.comm);

        int nnodes = crosscomms[nfields-1].size;
        int nL1    = crosscomms[nfields-2].size;
        MPI_Allreduce(MPI_IN_PLACE, &nnodes, 1, MPI_INT, MPI_MAX, world.comm);
        MPI_Allreduce(MPI_IN_PLACE, &nL1,    1, MPI_INT, MPI_MAX, world.comm);

        int max_ppn, ppn = subcomms[nfields-1].size;
        MPI_Allreduce(&ppn, &max_ppn, 1, MPI_INT, MPI_MAX, world.comm);

        if (world.rank == 0) fprintf(stdout, "[%d] max_ppn: %d, max_nps: %d, nnodes: %d, nL1: %d\n", world.rank, max_ppn, max_nps, nnodes, nL1);

        int *L1_size = (int*)malloc(sizeof(int)*(nfields-1));
        L1_size[nfields-2] = 1;
        for (int i=nfields-3; i>=0; i--) {
            if (subcrosscomms[i+1].comm != MPI_COMM_NULL)
                MPI_Reduce(&L1_size[i+1], &L1_size[i], 1, MPI_INT, MPI_SUM, 0, subcrosscomms[i+1].comm);
            MPI_Bcast(&L1_size[i], 1, MPI_INT, 0, subcomms[i].comm);
        }


        auto *graph_switch = new std::vector<std::pair<char*,int>>[nfields];
        auto *graph_nodes  = new std::vector<std::pair<char*,int>>[nL1];
        std::vector<std::pair<char*,int>> graph_tmp_1;
        std::vector<std::pair<char*,int>> graph_tmp_2;
        for(int level=0; level<nfields; level++) {
            if (crosscomms[level].comm != MPI_COMM_NULL) {
                char s[50];
                sprintf(s, "[%d] Names at level (%d)", world.rank, level);
                int    str_len = (level != nfields-1) ? addr.switch_name_lens[level] : addr.node_name_len ;
                char  *str     = (level != nfields-1) ? addr.switch_names[level]     : addr.node_name ;
                char **tmp     = allgather_strings(str_len, str, crosscomms[level].comm);

                int  togather = (level!=nfields-1) ? L1_size[level] : ppn ;
                int *sizes = (int*)malloc(sizeof(int)*crosscomms[level].size);
                MPI_Allgather(&togather, 1, MPI_INT, sizes, 1, MPI_INT, crosscomms[level].comm);

                std::vector<std::pair<char*,int>>& g = (level != nfields-1) ? graph_switch[level] : graph_tmp_2 ;
                for (int i=0; i<crosscomms[level].size; i++) g.push_back(std::make_pair(tmp[i],sizes[i]));

                if (level == nfields-2) {
                    int *nps_vec = (int*)malloc(sizeof(int)*crosscomms[level].size);
                    MPI_Allgather(&nps, 1, MPI_INT, nps_vec, 1, MPI_INT, crosscomms[level].comm);
                    for (int i=0; i<crosscomms[level].size; i++) graph_tmp_1.push_back(std::make_pair(tmp[i],nps_vec[i]));
                    free(nps_vec);
                }

                // for (int i=0; i<crosscomms[level].size; i++) free(tmp[i]); // NOTE: not to free, in graph_switch there's just a pointer
                free(sizes);
                free(tmp);
            }
        }

        if (world.rank == 0) {
            std::cout << "graph_tmp_1: ";
            for (std::pair<char*,int> e: graph_tmp_1) std::cout << '{' << std::get<0>(e) << ',' << std::get<1>(e) << "}," ;
            std::cout << std::endl;

            std::cout << "graph_tmp_2: ";
            for (std::pair<char*,int> e: graph_tmp_2) std::cout << '{' << std::get<0>(e) << ',' << std::get<1>(e) << "}," ;
            std::cout << std::endl;
        }


        int i = 0, k = 0;
        for (std::pair<char*,int> e: graph_tmp_1) {
            for (int j=0; j<std::get<1>(e); j++) {
                graph_nodes[i].push_back(graph_tmp_2[k]);
                k++;
            }
            i++;
        }

        // Populate the network graph
        graph.nl1     = nL1;        // Number of L1 switches (i.e. lower level switches)
        graph.shight  = nfields-1;  // High of the switch tree
        graph.max_nps = max_nps;    // Max number of nodes per L1 switch
        graph.max_ppn = ppn;        // Max number of processes per node

        graph.gswitch = graph_switch;
        graph.gnodes  = graph_nodes;

        MPI_Allreduce(&addr.max_lable_len, &graph.max_lable_len, 1, MPI_INT, MPI_MAX, world.comm);

        graph.node_width = (3*max_ppn > graph.max_lable_len) ? 3*max_ppn : graph.max_lable_len ;
        graph.l1_width   = graph.node_width +2;
    }
};


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


    // --------------------- New ADDR ---------------------
    // const char *addr_str = simulate_addr_str(rank, size);
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

    // MPI_Comm new_world, *subcomms, *crosscomms, *subcrosscomm;
    // new_world  = newworld_buffers.gen_new_world();
    // subcomms   = newworld_buffers.gen_sub_comms();
    // crosscomms = newworld_buffers.gen_cross_comms();
    // subcrosscomm = newworld_buffers.subcrosscomms;
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

/*
    sleep(1);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int j=0; j<addr.nswitchs+1; j++) {

        int tmp_rank, tmp_size;
        MPI_Comm tmp_comm = subcomms[j];
        MPI_Comm_size(tmp_comm, &tmp_size);
        MPI_Comm_rank(tmp_comm, &tmp_rank);

        int ngroups = newworld_buffers.ord_vec[j].size();
        if (rank == 0) fprintf(stdout, "--------------- Subcomm %d ---------------\n", j); fflush(stdout);

        for (int k=0; k<ngroups; k++) {
            if (newworld_buffers.subcomm_colors[j] == k) {
                if (tmp_rank == 0) fprintf(stdout, "---- Group %d ----\n", k); fflush(stdout);
                for (int i=0; i<tmp_size; i++) {
                    if (i == tmp_rank) {
                        fprintf(stdout, "[%2d] ADDR string: %s (%d)\n", new_rank, addr.addr_str, rank);
                    }
                    fflush(stdout);
                    MPI_Barrier(tmp_comm);
                    usleep(250);
                }
            }
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
            usleep(250);
        }
    }

    for (int j=0; j<addr.nswitchs+1; j++) {

        if (rank == 0) fprintf(stdout, "--------------- Crosscomm %d ---------------\n", j); fflush(stdout);

        if (crosscomms[j] != MPI_COMM_NULL) {
            int cross_rank, cross_size;
            MPI_Comm cross_comm = crosscomms[j];
            MPI_Comm_size(cross_comm, &cross_size);
            MPI_Comm_rank(cross_comm, &cross_rank);

            int* gatheredvec = (int*)malloc(sizeof(int)*cross_size);
            MPI_Allgather(&new_rank, 1, MPI_INT, gatheredvec, 1, MPI_INT, cross_comm);
            if (cross_rank == 0) {
                fprintf(stdout, "[%d] at level %d: ", new_rank, j);
                print_array_inline(gatheredvec, cross_size);
            }
        }

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(500);
    }

    for (int j=0; j<addr.nswitchs+1; j++) {

        if (rank == 0) fprintf(stdout, "--------------- Subcrosscomm %d ---------------\n", j); fflush(stdout);

        if (newworld_buffers.subcrosscomms[j] != MPI_COMM_NULL) {
            int cross_rank, cross_size;
            MPI_Comm cross_comm = newworld_buffers.subcrosscomms[j];
            MPI_Comm_size(cross_comm, &cross_size);
            MPI_Comm_rank(cross_comm, &cross_rank);

            int* gatheredvec = (int*)malloc(sizeof(int)*cross_size);
            MPI_Allgather(&new_rank, 1, MPI_INT, gatheredvec, 1, MPI_INT, cross_comm);
            if (cross_rank == 0) {
                fprintf(stdout, "[%d] at level %d: ", new_rank, j);
                print_array_inline(gatheredvec, cross_size);
            }
        }

        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(500);
    }

    if (rank == 0) fprintf(stdout, "-------------------------------------\n");
*/
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
