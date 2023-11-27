#pragma once

#define XSTR(S) #S
#define STR_CONCAT(S1, S2, S3) XSTR(S1) XSTR(S2) XSTR(S3)

#define AC_BLACK \x1b[30m
#define AC_RED \x1b[31m
#define AC_GREEN \x1b[32m
#define AC_YELLOW \x1b[33m
#define AC_BLUE \x1b[34m
#define AC_MAGENTA \x1b[35m
#define AC_CYAN \x1b[36m
#define AC_WHITE \x1b[37m
#define AC_NORMAL \x1b[m

#define COLOUR(C, TX) STR_CONCAT(C, TX, AC_NORMAL)

#ifdef DEBUG
#define DBG_CHECK(X) { \
    if( X == DEBUG ) {  \
      fprintf(stdout, "%s: file %s at line %d\n", COLOUR(AC_CYAN, DEBUG_CHECK), __FILE__, __LINE__ ); \
      fflush(stdout); \
    }  \
  }
#define MPI_DBG_CHECK(X) { \
    if( X == DEBUG ) {  \
      fprintf(stdout, "%s[%d]: file %s at line %d\n", COLOUR(AC_CYAN, DEBUG_CHECK), gmyid, __FILE__, __LINE__ ); \
      fflush(stdout); \
    }  \
  }
#define DBG_PRINT(X, Y) { \
    if ( X == DEBUG ) {  \
      Y  \
      fflush(stdout);  \
    }  \
  }
#define DBG_STOP(X) {      \
    if ( X == DEBUG ) {    \
      fprintf(stderr, "%s invoked at line %d of file %s\n", COLOUR(AC_RED, DEBUG_STOP), __LINE__, __FILE__); \
      fflush(stderr);      \
      exit(42);            \
    }                      \
  }
#else
#define DBG_CHECK(X) {  }
#define MPI_DBG_CHECK(X) {  }
#define DBG_PRINT(X, Y) { }
#define DBG_STOP(X) { }
#endif

#define MPI_ALL_PRINT(X) \
  {\
    FILE *fp;\
    char s[50], s1[50];\
    sprintf(s, "temp_%d.txt", myid);\
    fp = fopen ( s, "w" );\
    fclose(fp);\
    fp = fopen ( s, "a+" );\
    fprintf(fp, "\t------------------------- Proc %d File %s Line %d -------------------------\n\n", myid, __FILE__, __LINE__);\
    X;\
    if (myid==ntask-1) \
        fprintf(fp, "\t--------------------------------------------------------------------------\n\n");\
    fclose(fp);\
    for (int i=0; i<ntask; i++) {\
        if (myid == i) {\
            int error; \
            sprintf(s1, "cat temp_%d.txt", myid);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
            sprintf(s1, "rm temp_%d.txt", myid);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
        }\
        MPI_Barrier(MPI_COMM_WORLD);\
    }\
  }

#define MPI_COMMUNICATOR_PRINT(CM, X)  \
  {\
    int inmacro_myid, inmacro_ntask;  \
    MPI_Comm_rank(CM, &inmacro_myid);  \
	MPI_Comm_size(CM, &inmacro_ntask);  \
    FILE *fp;\
    char s[50], s1[50];\
    sprintf(s, "temp_%d.txt", inmacro_myid);\
    fp = fopen ( s, "w" );\
    fclose(fp);\
    fp = fopen ( s, "a+" );\
    fprintf(fp, "\t------------------------- Proc %d File %s Line %d -------------------------\n\n", inmacro_myid, __FILE__, __LINE__);\
    X;\
    if (inmacro_myid==inmacro_ntask-1) \
        fprintf(fp, "\t--------------------------------------------------------------------------\n\n");\
    fclose(fp);\
    for (int i=0; i<inmacro_ntask; i++) {\
        if (inmacro_myid == i) {\
            int error; \
            sprintf(s1, "cat temp_%d.txt", inmacro_myid);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
            sprintf(s1, "rm temp_%d.txt", inmacro_myid);\
            error = system(s1);\
            if (error == -1) fprintf(stderr, "Error at line %d of file %s", __LINE__, __FILE__); \
        }\
        MPI_Barrier(CM);\
    }\
  }

#define MPI_PROCESS_PRINT(CM, P, X)  \
  {\
    int myid, ntask;  \
    MPI_Comm_rank(CM, &myid);  \
	MPI_Comm_size(CM, &ntask);  \
	if (myid == P) {  \
      fprintf(stdout, "\t--------------------- Proc %d of %d. File %s Line %d ---------------------\n\n", myid, ntask, __FILE__, __LINE__);\
      X;\
      fprintf(stdout, "\t--------------------------------------------------------------------------\n\n");\
    }  \
  }

#define OUTOFBOUNDS_NUMBER(FP, P) {                         \
    unsigned long long int i = 0;                           \
    if ((void*)P != NULL) {                                 \
      while ((void*)(P + i) != NULL) {                      \
        i++;                                                \
      }                                                     \
    } else {                                                \
      fprintf(FP, "%s is NULL\n", #P);                      \
    }                                                       \
    fprintf(FP, "The array lenght of %s is %llu\n", #P, i); \
  }

#define PRINT_VECTOR( V, LN, NM, FP ) {     \
    fprintf(FP, "%2s: ", NM);               \
    for (int I=0; I<LN; I++)                \
        fprintf(FP, "%4d ", V[I]);          \
    fprintf(FP, "\n");                      \
}

#define PRINT_FLOAT_VECTOR( V, LN, NM, FP ) {  \
    fprintf(FP, "%2s: ", NM);                  \
    for (int I=0; I<LN; I++)                   \
        fprintf(FP, "%.3f ", V[I]);            \
    fprintf(FP, "\n");                         \
}

#define PRINT_MATRIX( A, N, M) {                        \
    for (int i_macro=0; i_macro<(N); i_macro++) {       \
        for (int j_macro=0; j_macro<(M); j_macro++)     \
            printf("%3.2f ", A[i_macro*(M) + j_macro]); \
        printf("\n");                                   \
    }                                                   \
    printf("\n");                                       \
}

#define PRINT_CSR( RW, CL, VL, N, M, NZ, NM ) {       \
    printf("Sparse matrix %s:\n", (NM));              \
    printf("%s->n = %d\n", (NM), (N));                \
    printf("%s->m = %d\n", (NM), (M));                \
    printf("%s->nnz = %d\n", (NM), (NZ));             \
    printf("Rows: ");                                 \
    for (int i_macro=0; i_macro<((N)+1); i_macro++)   \
        printf("%3d ", RW[i_macro]);                  \
    printf("\n");                                     \
    printf("Cols: ");                                 \
    for (int i_macro=0; i_macro<(NZ); i_macro++)      \
        printf("%3d ", CL[i_macro]);                  \
    printf("\n");                                     \
    printf("Vals: ");                                 \
    for (int i_macro=0; i_macro<(NZ); i_macro++)      \
        printf("%3.2f ", VL[i_macro]);                \
    printf("\n\n");                                   \
}

#define PRINT_CSC( RW, CL, VL, N, M, NZ, NM ) {       \
    printf("Sparse matrix %s:\n", (NM));              \
    printf("%s->n = %d\n", (NM), (N));                \
    printf("%s->m = %d\n", (NM), (M));                \
    printf("%s->nnz = %d\n", (NM), (NZ));             \
    printf("Rows: ");                                 \
    for (int i_macro=0; i_macro<(NZ); i_macro++)      \
        printf("%3d ", RW[i_macro]);                  \
    printf("\n");                                     \
    printf("Cols: ");                                 \
    for (int i_macro=0; i_macro<((M)+1); i_macro++)   \
        printf("%3d ", CL[i_macro]);                  \
    printf("\n");                                     \
    printf("Vals: ");                                 \
    for (int i_macro=0; i_macro<(NZ); i_macro++)      \
        printf("%3.2f ", VL[i_macro]);                \
    printf("\n\n");                                   \
}

#define PRINT_NOVAL_CSC( RW, CL, N, M, NZ, NM, FP ) { \
    fprintf((FP), "Sparse matrix %s:\n", (NM));       \
    fprintf((FP), "%s->n = %d\n", (NM), (N));         \
    fprintf((FP), "%s->m = %d\n", (NM), (M));         \
    fprintf((FP), "%s->nnz = %d\n", (NM), (NZ));      \
    fprintf((FP), "Rows: ");                          \
    for (int i_macro=0; i_macro<(NZ); i_macro++)      \
        fprintf((FP), "%3d ", RW[i_macro]);           \
    fprintf((FP), "\n");                              \
    fprintf((FP), "Cols: ");                          \
    for (int i_macro=0; i_macro<((M)+1); i_macro++)   \
        fprintf((FP), "%3d ", CL[i_macro]);           \
    fprintf((FP), "\n\n");                            \
}

#ifdef NCCL

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#endif
