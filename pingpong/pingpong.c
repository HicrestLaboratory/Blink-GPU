// Copyright 2009-2018 Sandia Corporation. Under the terms
// of Contract DE-NA0003525 with Sandia Corporation, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2018, Sandia Corporation
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define PINGPONG_REPEATS 1000
#define PINGPONG_MSG_SIZE 1024

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int me = -1;
  int world = -1;

  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  int msgSize = PINGPONG_MSG_SIZE;
  int repeats = PINGPONG_REPEATS;

  if (argc > 1) {
    msgSize = atoi(argv[1]);
  }

  if (argc > 2) {
    repeats = atoi(argv[2]);
  }

  if (0 == me) {
    printf("# MPI PingPong Pattern\n");
    printf("# Info:\n");
    printf("# - Total Ranks:     %10d\n", world);
    printf("# - Message Size:    %10d Bytes\n", msgSize);
    printf("# - Repeats:         %10d\n", repeats);
  }

  if (world < 2) {
    printf("No MPI is run because there are not 2 or more processors.\n");
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (me < 2) {
    char* sendBuffer = (char*)malloc(sizeof(char*) * msgSize);
    char* recvBuffer = (char*)malloc(sizeof(char*) * msgSize);

    MPI_Status status;

    for (int i = 0; i < msgSize; ++i) {
      sendBuffer[i] = i;
      recvBuffer[i] = 0;
    }

    if (0 == me) {
      printf("# Beginning benchmarking...\n");
    }

    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    for (int i = 0; i < repeats; ++i) {
      if (0 == me) {
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
      } else {
        MPI_Recv(recvBuffer, msgSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(sendBuffer, msgSize, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
      }
    }

    gettimeofday(&end, NULL);

    free(sendBuffer);
    free(recvBuffer);

    if (0 == me) {
      printf("# Statistics:\n");

      const double bytesXchng = ((double)msgSize) * 2.0 * ((double)repeats);
      const double MbytesXchng = bytesXchng / (1024.0 * 1024.0);
      const double timeTaken =
          (((double)end.tv_sec) + 1.0e-6 * ((double)end.tv_usec)) -
          (((double)start.tv_sec) + 1.0e-6 * ((double)start.tv_usec));
      const double msgsXchng = ((double)repeats) * 2.0;
      const double KMsgsXchng = msgsXchng / 1000.0;

      printf("# %9s %11s %17s %14s %16s %14s\n", "MsgSize", "Time", "KMsgs",
             "MB", "KMsg/S", "MB/S");
      printf("  %9.0f %11.4f %17.5f %14.4f %16.4f %14.4f\n", (double)msgSize,
             timeTaken, KMsgsXchng, MbytesXchng, KMsgsXchng / timeTaken,
             MbytesXchng / timeTaken);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
