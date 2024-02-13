#pragma once

// ========================================== DBG UTILS ===========================================

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
#define MPI_DBG_CHECK(CM, X) {                                                                                          \
    int inmacro_myid;                                                                                                   \
    MPI_Comm_rank(CM, &inmacro_myid);                                                                                   \
    if( X == DEBUG ) {                                                                                                  \
      fprintf(stdout, "%s[%d]: file %s at line %d\n", COLOUR(AC_CYAN, DEBUG_CHECK), inmacro_myid, __FILE__, __LINE__ ); \
      fflush(stdout);                                                                                                   \
    }                                                                                                                   \
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

#define HEAP_TRACKER_BLK 10

struct my_heap_tracker {
  size_t  len;
  size_t  size;
  void**  ptr_list;
  size_t* alloc_list;
  size_t* free_list;
};

struct my_heap_tracker myHeapTracker;

#define HEAP_TRACKER_OPEN { \
  myHeapTracker.len = 0; \
  myHeapTracker.size = HEAP_TRACKER_BLK; \
  myHeapTracker.ptr_list = malloc(sizeof(void*)*HEAP_TRACKER_BLK); \
  myHeapTracker.free_list = malloc(sizeof(size_t*)*HEAP_TRACKER_BLK); \
  myHeapTracker.alloc_list = malloc(sizeof(size_t*)*HEAP_TRACKER_BLK); \
  for (int i=0; i<HEAP_TRACKER_BLK; i++) { \
    myHeapTracker.ptr_list[i] = NULL; \
    myHeapTracker.free_list[i] = 0; \
    myHeapTracker.alloc_list[i] = 0; \
  } \
}

#define MY_MALLOC_CHECK(P, X) {\
  void* p = X;\
  if (p == NULL) {\
    fprintf(stderr, "ERROR: malloc at line %d of file %s reeturned a NULL pointer\n", __LINE__, __FILE__);\
    exit();\
  }\
\
  P = p;\
}

#else
#define DBG_CHECK(X) {  }
#define MPI_DBG_CHECK(CM, X) {  }
#define DBG_PRINT(X, Y) { }
#define DBG_STOP(X) { }
#define MY_MALLOC_CHECK(X) X;
#endif

// ========================================== MPI PRINTS ==========================================

#define MPI_ALL_PRINT(X) {                                                                                                              \
    int inmacro_myid, inmacro_ntask;                                                                                                    \
    MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                                                                       \
	MPI_Comm_size(MPI_COMM_WORLD, &inmacro_ntask);                                                                                      \
	\
    FILE *fp;                                                                                                                           \
    char s[50], s1[50];                                                                                                                 \
    sprintf(s, "temp_%d.txt", inmacro_myid);                                                                                            \
    fp = fopen ( s, "w" );                                                                                                              \
    fclose(fp);                                                                                                                         \
    fp = fopen ( s, "a+" );                                                                                                             \
    fprintf(fp, "\t------------------------- Proc %d File %s Line %d -------------------------\n\n", inmacro_myid, __FILE__, __LINE__); \
    X;                                                                                                                                  \
    if (inmacro_myid==inmacro_ntask-1)                                                                                                  \
        fprintf(fp, "\t--------------------------------------------------------------------------\n\n");                                \
    fclose(fp);                                                                                                                         \
    \
    for (int i=0; i<inmacro_ntask; i++) {                                                                                               \
        if (inmacro_myid == i) {                                                                                                        \
            int error;                                                                                                                  \
            sprintf(s1, "cat temp_%d.txt", inmacro_myid);                                                                               \
            error = system(s1);                                                                                                         \
            if (error == -1) fprintf(stderr, "Error at line %d of file %s\n", __LINE__, __FILE__);                                      \
            sprintf(s1, "rm temp_%d.txt", inmacro_myid);                                                                                \
            error = system(s1);                                                                                                         \
            if (error == -1) fprintf(stderr, "Error at line %d of file %s\n", __LINE__, __FILE__);                                      \
        }                                                                                                                               \
        MPI_Barrier(MPI_COMM_WORLD);                                                                                                    \
    }                                                                                                                                   \
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

// ======================================= STRING COLLECTOR =======================================

#define STR_COLL_BLK 1024

struct string_collector {
    char *collector = NULL;
    char *buff  = NULL;
    int size = 0;
    int len = 0;
};

#define STR_COLL_DEF struct string_collector str_coll;

#define STR_COLL_INIT {                                           \
  str_coll.collector = (char*)malloc(sizeof(char)*STR_COLL_BLK);  \
  str_coll.buff = (char*)malloc(sizeof(char)*STR_COLL_BLK);       \
  str_coll.collector[0]='\0';                                     \
  str_coll.size = STR_COLL_BLK;                                   \
  str_coll.len  = 0;                                              \
}

#define STR_COLL_APPEND(X) {                                                              \
  X                                                                                       \
  int n = strlen(str_coll.buff);                                                          \
  if (str_coll.len + n >= str_coll.size) {                                                \
    str_coll.collector = (char*)realloc(str_coll.collector, str_coll.size + STR_COLL_BLK);\
    str_coll.size += STR_COLL_BLK;                                                        \
  }                                                                                       \
  memcpy(&str_coll.collector[str_coll.len], str_coll.buff, sizeof(char)*n);               \
  str_coll.len += n;                                                                      \
  str_coll.collector[str_coll.len]='\0';                                                  \
  memset(str_coll.buff, 0, sizeof(char) * STR_COLL_BLK);                                  \
}

#define STR_COLL_BUFF str_coll.buff
#define STR_COLL_GIVE str_coll.collector

#define STR_COLL_FREE {       \
    free(str_coll.collector); \
    free(str_coll.buff);      \
    str_coll.size = 0;        \
    str_coll.len = 0;         \
}

#define MPI_ALL_PRINT2(X) {                                                                                                             \
    fflush(stdout);                                                                                                                     \
    MPI_Barrier(MPI_COMM_WORLD);                                                                                                        \
    int inmacro_myid, inmacro_ntask;                                                                                                    \
    MPI_Comm_rank(MPI_COMM_WORLD, &inmacro_myid);                                                                                       \
	MPI_Comm_size(MPI_COMM_WORLD, &inmacro_ntask);                                                                                      \
                                                                                                                                        \
    for (int i=0; i<inmacro_ntask; i++) {                                                                                               \
        if (inmacro_myid == i) {                                                                                                        \
          printf("\t------------------------- Proc %d File %s Line %d -------------------------\n\n", inmacro_myid, __FILE__, __LINE__);\
          X                                                                                                                             \
          if (inmacro_myid==inmacro_ntask-1)                                                                                            \
            printf("\t--------------------------------------------------------------------------\n\n");                                 \
          fflush(stdout);                                                                                                               \
        }                                                                                                                               \
        MPI_Barrier(MPI_COMM_WORLD);                                                                                                    \
    }                                                                                                                                   \
  }

// ================================================================================================

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

// ==================================== VECTOR & MATRIX PRINT =====================================

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
            printf("%d ", A[i_macro*(M) + j_macro]);    \
        printf("\n");                                   \
    }                                                   \
    printf("\n");                                       \
}

#define FPRINT_MATRIX( FP, A, N, M) {                     \
    for (int i_macro=0; i_macro<(N); i_macro++) {         \
        for (int j_macro=0; j_macro<(M); j_macro++)       \
            fprintf(FP, "%d ", A[i_macro*(M) + j_macro]); \
        fprintf(FP, "\n");                                \
    }                                                     \
    fprintf(FP, "\n");                                    \
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

// ================================================================================================

#define CUDA // tmp BUG

#ifdef CUDA
// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaResult(err)  __checkCudaResult (err, __FILE__, __LINE__)

inline void __checkCudaResult( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

#include "helper_cuda.h"

struct dev_vec_collector {
  size_t len;
  size_t size;

  size_t max_size;
  size_t *vec_sizes;
  float  **vec_ptrs;
  char   **vec_names;

  float  *cpu_buff;
  size_t cpu_buff_len;
  size_t cpu_buff_size;
  char   *cpu_buff_name;

//   size_t dev_coll_len;
//   size_t *dev_vec_sizes;
//   float  **dev_vec_ptrs;
};

// BUG BUG BUG with realloc
#define DVC_BLK 10

#define DEF_DVC struct dev_vec_collector dev_vec_coll;

#define INIT_DVC {                                                    \
  dev_vec_coll.len  = 0;                                              \
  dev_vec_coll.size = DVC_BLK;                                        \
                                                                      \
  dev_vec_coll.max_size  = 0;                                         \
  dev_vec_coll.vec_sizes = (size_t*) malloc(sizeof(size_t)*DVC_BLK);  \
  dev_vec_coll.vec_ptrs  = (float**) malloc(sizeof(float*)*DVC_BLK);  \
  dev_vec_coll.vec_names = (char**)  malloc(sizeof(char*)*DVC_BLK);   \
                                                                      \
  dev_vec_coll.cpu_buff      = NULL;                                  \
  dev_vec_coll.cpu_buff_len  = 0;                                     \
  dev_vec_coll.cpu_buff_size = 0;                                     \
  dev_vec_coll.cpu_buff_name = NULL;                                  \
}

//                                                                       \
//   dev_vec_coll.dev_coll_len  = 0;                                     \
//   dev_vec_coll.dev_vec_sizes = NULL;                                  \
//   dev_vec_coll.dev_vec_ptrs  = NULL;                                  \

#define APPEND_DVC(V, S, N) {                                                                         \
  if (dev_vec_coll.len == dev_vec_coll.size-1) {                                                      \
    dev_vec_coll.vec_sizes = (size_t*) realloc(dev_vec_coll.vec_sizes, dev_vec_coll.size + DVC_BLK);  \
    dev_vec_coll.vec_ptrs  = (float**) realloc(dev_vec_coll.vec_ptrs,  dev_vec_coll.size + DVC_BLK);  \
    dev_vec_coll.vec_names = (char**)  realloc(dev_vec_coll.vec_names,  dev_vec_coll.size + DVC_BLK); \
    dev_vec_coll.size += DVC_BLK;                                                                     \
  }                                                                                                   \
  dev_vec_coll.vec_ptrs[dev_vec_coll.len]  = V;                                                       \
  dev_vec_coll.vec_sizes[dev_vec_coll.len] = S;                                                       \
  dev_vec_coll.vec_names[dev_vec_coll.len] = (char*)N;                                                \
  if (S > dev_vec_coll.max_size)                                                                      \
    dev_vec_coll.max_size = S;                                                                        \
  dev_vec_coll.len++;                                                                                 \
}

#define DVC_LEN dev_vec_coll.len
#define DVC_MAXSIZE dev_vec_coll.max_size

#define DVC_TOCPU(I) {                                                                                \
  if (dev_vec_coll.len > 0 && I < dev_vec_coll.len) {                                                 \
    if (dev_vec_coll.cpu_buff_size == 0) {                                                            \
      dev_vec_coll.cpu_buff  = (float*) malloc(sizeof(float)*dev_vec_coll.max_size);                  \
      dev_vec_coll.cpu_buff_size = dev_vec_coll.max_size;                                             \
    } else if (dev_vec_coll.cpu_buff_size < dev_vec_coll.max_size) {                                  \
      dev_vec_coll.cpu_buff  = (float*) realloc(dev_vec_coll.cpu_buff, dev_vec_coll.max_size);        \
      dev_vec_coll.cpu_buff_size = dev_vec_coll.max_size;                                             \
    }                                                                                                 \
                                                                                                      \
    checkCudaErrors( cudaMemcpy(dev_vec_coll.cpu_buff, dev_vec_coll.vec_ptrs[I], dev_vec_coll.vec_sizes[I]*sizeof(float), cudaMemcpyDeviceToHost) );                                                                        \
    dev_vec_coll.cpu_buff_len = dev_vec_coll.vec_sizes[I];                                            \
    dev_vec_coll.cpu_buff_name = dev_vec_coll.vec_names[I];                                           \
    checkCudaErrors( cudaDeviceSynchronize() );                                                       \
  } else {                                                                                            \
    fprintf(stderr, "Error at line %d of file %s invoked by MACRO ...\n", __LINE__, __FILE__);        \
  }                                                                                                   \
}

#define DVC_CPUBUFF dev_vec_coll.cpu_buff
#define DVC_CPUBUFFLEN dev_vec_coll.cpu_buff_len
#define DVC_CPUBUFFNAM dev_vec_coll.cpu_buff_name

// #define DVC_TOGPU { \
//   if (dev_vec_coll.dev_coll_len == 0) {\
//     checkCudaErrors( cudaMalloc(&dev_vec_coll.dev_vec_sizes,  sizeof(size_t) * dev_vec_coll.len) );\
//     checkCudaErrors( cudaMalloc(&dev_vec_coll.dev_vec_ptrs,   sizeof(float*) * dev_vec_coll.len) );\
//     dev_vec_coll.dev_coll_len = dev_vec_coll.len;\
//   } else if (dev_vec_coll.dev_coll_len < dev_vec_coll.len) {\
//     checkCudaErrors( cudaFree(dev_vec_coll.dev_vec_sizes) );  \
//     checkCudaErrors( cudaFree(dev_vec_coll.dev_vec_ptrs) );   \
//     checkCudaErrors( cudaMalloc(&dev_vec_coll.dev_vec_sizes,  sizeof(size_t) * dev_vec_coll.len) );\
//     checkCudaErrors( cudaMalloc(&dev_vec_coll.dev_vec_ptrs,   sizeof(float*) * dev_vec_coll.len) );\
//     dev_vec_coll.dev_coll_len = dev_vec_coll.len;\
//   }\
// \
//   checkCudaErrors( cudaMemcpy(dev_vec_coll.dev_vec_sizes,  dev_vec_coll.vec_sizes,  sizeof(size_t) * dev_vec_coll.len, cudaMemcpyHostToDevice) );\
//   checkCudaErrors( cudaMemcpy(dev_vec_coll.dev_vec_ptrs, dev_vec_coll.vec_ptrs,   sizeof(float*) * dev_vec_coll.len, cudaMemcpyHostToDevice) );\
// }

// #define DVC_GPUCOLL dev_vec_coll.dev_vec_ptrs
// #define DVC_GPUSIZES dev_vec_coll.dev_vec_sizes

#define FREE_DVC {                    \
  free(dev_vec_coll.vec_sizes);       \
  free(dev_vec_coll.vec_names);       \
  free(dev_vec_coll.vec_ptrs);        \
  if (dev_vec_coll.cpu_buff_len != 0) \
    free(dev_vec_coll.cpu_buff);      \
                                      \
  dev_vec_coll.len = 0;               \
  dev_vec_coll.size = 0;              \
  dev_vec_coll.max_size = 0;          \
  dev_vec_coll.cpu_buff_len = 0;      \
  dev_vec_coll.cpu_buff_name = NULL;  \
}

//   if (dev_vec_coll.dev_coll_len != 0) {                       \
//     checkCudaErrors( cudaFree(dev_vec_coll.dev_vec_sizes) );  \
//     checkCudaErrors( cudaFree(dev_vec_coll.dev_vec_ptrs) );   \
//   }                                                           \


// #define PRINT_DEV_VEC_ENTRANCE(CM, N) {
//   DEF_DVC
//   INIT_DVC
//
//   srand((unsigned int)time(NULL));
//   int x = rand() % (GRD_SIZE*BLK_SIZE);
//
//   APPEND_DVC(dev_xSendBuffer, xSize, "dev_xSendBuffer")
//   APPEND_DVC(dev_ySendBuffer, ySize, "dev_ySendBuffer")
//
//   MPI_COMMUNICATOR_PRINT(CM,
//     fprintf(fp, "extracted tid = %d\n", x);
//     for (int i=0; i<N; i++) {
//       DVC_TOCPU(i)
//       fprintf(fp, "%s = %6.4f\n", DVC_CPUBUFFNAM, DVC_CPUBUFF[x]);
//     }
//   )
//   FREE_DVC
//   MPI_Barrier(CM);
// }

#endif

int cmpfunc (const void * a, const void * b) {
   if(*(float*)a == *(float*)b) {return (0);} else {
   return ( (*(float*)a > *(float*)b) ? 1 : -1 ); }
}

#define PRINT_STREAM_TIMETABLE(M, N) {                \
  float in_macro_tmp[N];                              \
  printf("\nStart:\t");                               \
  for (int i=0; i<N; i++) in_macro_tmp[i] = M[0][i];  \
  qsort(in_macro_tmp, N, sizeof(float), cmpfunc);     \
  for (int i=0; i<N; i++) {                           \
    int j=0;                                          \
    while (j<N && in_macro_tmp[i] != M[0][j]) j++;    \
    printf("%2d (%4.3f) |", j, in_macro_tmp[i]);      \
  }                                                   \
  printf("\n");                                       \
  \
  printf("Stop: \t");                                 \
  for (int i=0; i<N; i++) in_macro_tmp[i] = M[1][i];  \
  qsort(in_macro_tmp, N, sizeof(float), cmpfunc);     \
  for (int i=0; i<N; i++) {                           \
    int j=0;                                          \
    while (j<N && in_macro_tmp[i] != M[1][j]) j++;    \
    printf("%2d (%4.3f) |", j, in_macro_tmp[i]);      \
  }                                                   \
  printf("\n\nTime of each stream:\n");               \
  for (int i=0; i<N; i++) {                           \
    printf("\t Stream %d:%4.3f\n",i,M[1][i]-M[0][i]); \
  }                                                   \
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
