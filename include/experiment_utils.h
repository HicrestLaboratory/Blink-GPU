#pragma once

#define TIMER_DEF(n)	 struct timeval temp_1_##n={0,0}, temp_2_##n={0,0}
#define TIMER_START(n)	 gettimeofday(&temp_1_##n, (struct timezone*)0)
#define TIMER_STOP(n)	 gettimeofday(&temp_2_##n, (struct timezone*)0)
#define TIMER_ELAPSED(n) ((temp_2_##n.tv_sec-temp_1_##n.tv_sec)+(temp_2_##n.tv_usec-temp_1_##n.tv_usec)*1.0e-6)

#define NEXP 8
#define MAX_EXPLABLE_SIZE 10

typedef struct experiments_statistics {
    char* expname[NEXP];
    char* exptype[NEXP];
    char*  layout[NEXP];
    char*  lables[NEXP];
    double  error[NEXP];
    double  times[NEXP];
    unsigned long long int int_error[NEXP];

    int num_round[NEXP];
    double avg_time[NEXP];
    double std_time[NEXP];

} ExperimentsStatistics;

ExperimentsStatistics exp_stats;

#define INIT_EXPS {                                                         \
    for (int i=0; i<NEXP; i++) {                                            \
        char *tmp0 = (char*)malloc(sizeof(char)*MAX_EXPLABLE_SIZE);         \
        char *tmp1 = (char*)malloc(sizeof(char)*MAX_EXPLABLE_SIZE);         \
        char *tmp2 = (char*)malloc(sizeof(char)*MAX_EXPLABLE_SIZE);         \
        char *tmp3 = (char*)malloc(sizeof(char)*MAX_EXPLABLE_SIZE);         \
        strcpy(tmp0, "-----");                                              \
        strcpy(tmp1, "-----");                                              \
        strcpy(tmp2, "-----");                                              \
        strcpy(tmp3, "-----");                                              \
        exp_stats.lables[i]     = tmp0;                                     \
        exp_stats.layout[i]     = tmp1;                                     \
        exp_stats.expname[i]    = tmp2;                                     \
        exp_stats.exptype[i]    = tmp3;                                     \
        exp_stats.error[i]      = 0.0;                                      \
        exp_stats.int_error[i]  = 0ULL;                                     \
        exp_stats.avg_time[i]   = 0.0;                                      \
        exp_stats.std_time[i]   = 0.0;                                      \
    }                                                                       \
}

#define SET_EXPERIMENT(I, LB) {                                                                                                 \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.lables[I] = (char*) LB ;                                                                                          \
}

#define SET_EXPERIMENT_NAME(I, LB) {                                                                                            \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.expname[I] = (char*) LB ;                                                                                         \
}

#define SET_EXPERIMENT_TYPE(I, LB) {                                                                                            \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.exptype[I] = (char*) LB ;                                                                                         \
}

#define SET_EXPERIMENT_LAYOUT(I, LB) {                                                                                          \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.layout[I] = (char*) LB ;                                                                                          \
}

#define ADD_TIME_EXPERIMENT(I, TM) {                                                                                            \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.times[I] += TM ;                                                                                                  \
}

#define ADD_ERROR_EXPERIMENT(I, ER) {                                                                                           \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.error[I] += ER ;                                                                                                  \
}

#define ADD_INTERROR_EXPERIMENT(I, ER) {                                                                                        \
    if ( I >= NEXP) {                                                                                                           \
        fprintf(stderr, "Invalid experiment number in %s line %d; I was %d when NEXP is %d\n", __FILE__, __LINE__, I, NEXP);    \
        exit(42);                                                                                                               \
    }                                                                                                                           \
    exp_stats.int_error[I] += ER ;                                                                                              \
}

#define PRINT_EXPARIMENT_STATS {                                                                            \
    printf("================== EXPERIMENT STATS ==================\n");                                     \
    printf("%10s %10s %10s %10s %10s %10s\n", "Experiment", "Type", "Layout", "Lable", "Time", "IntError"); \
    for (int i=0; i<NEXP; i++) {                                                                            \
        printf("%10s %10s %10s %10s %9.8lf %10llu\n", exp_stats.expname[i], exp_stats.exptype[i],           \
               exp_stats.layout[i], exp_stats.lables[i], exp_stats.times[i], exp_stats.int_error[i]);       \
    }                                                                                                       \
}
