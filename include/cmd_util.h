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


#define N_COMM_KIND 6
enum CommKind {
    WORLD,
    NODE,
    CROSS_NODE,
    TREE,
    CROSS_TREE,
    ROOT
};

typedef struct config
{
    // Debug parameeters
    int  verbose;

    // Repetitions and buffer size
    int loop_count;
    int buff_cycle;
    int fix_buff_size;
    int max_buff_size;

    // Process displacement
    int ppn;
    int npl;
    int tree_high;

    // Computed
    int nprocess;

    // Communicator
    CommKind selectedComm;

} Config;

void parse_args(int argc, char ** argv, Config * config)
{
    config->verbose       = 0;

    config->loop_count    = LOOP_COUNT;
    config->buff_cycle    = BUFF_CYCLE;
    config->fix_buff_size = 0;

    config->tree_high = 0;
    config->npl       = 1;
    config->ppn       = 1;

    config->selectedComm = WORLD;

    int flag_loop  = 0;
    int flag_buff  = 0;
    int flag_xbuff = 0;

    int inc = 2;
    for (int i=1; i<argc; i+=inc)
    {
        inc = 2;
        const char * argname = (argv[i]);

        if (!strcmp(argname, "--loop-count"))
        {
            config->loop_count = atoi(argv[i+1]);
            flag_loop = 1;
        }
        else if (!strcmp(argname, "--fix-buff"))
        {
            config->fix_buff_size = atoi(argv[i+1]);
            flag_xbuff = 1;
        }
        else if (!strcmp(argname, "--buff-cycle"))
        {
            config->buff_cycle = atoi(argv[i+1]);
            flag_buff = 1;
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
        else if (!strcmp(argname, "--selected-comm"))
        {
            config->selectedComm = static_cast<CommKind>(atoi(argv[i+1]));
        }
    }

    if (flag_buff && flag_xbuff) {
        fprintf(stdout, "Error: just one parameeter between --buff-cycle and --fix-buff must be set\n");
        exit(__LINE__);
    }


    config->max_buff_size = (flag_xbuff == 0) ? (config->buff_cycle) : (config->fix_buff_size + 1) ;
    if (flag_xbuff) {
        fprintf(stdout, "fix_buff_size is set as %d, loop_count is %d\n", config->fix_buff_size, config->loop_count);
    } else {
        fprintf(stdout, "buff_cycle is set as %d, loop_count is %d\n", config->buff_cycle, config->loop_count);
    }

    config->nprocess = (config->ppn) * (config->npl) * (1<<(config->tree_high-1));

}

void print_config(const Config *cfg, FILE *fp = stdout) {
    if (!fp || !cfg) {
        return;
    }

    fprintf(fp, "Config {\n");

    /* Debug parameters */
    fprintf(fp, "  verbose        : %d\n", cfg->verbose);

    /* Repetitions and buffer size */
    fprintf(fp, "  loop_count     : %d\n", cfg->loop_count);
    fprintf(fp, "  buff_cycle     : %d\n", cfg->buff_cycle);
    fprintf(fp, "  fix_buff_size  : %d\n", cfg->fix_buff_size);

    /* Process displacement */
    fprintf(fp, "  ppn            : %d\n", cfg->ppn);
    fprintf(fp, "  npl            : %d\n", cfg->npl);
    fprintf(fp, "  tree_high      : %d\n", cfg->tree_high);

    /* Computed */
    fprintf(fp, "  nprocess       : %d\n", cfg->nprocess);

    /* Communicator */
    fprintf(fp, "  selectedComm   : %d\n", (int)cfg->selectedComm);

    fprintf(fp, "}\n");
}

#endif

