MPI_HOME=/apps/openmpi-3.0.0--gcc-9.1.0

CC=${MPI_HOME}/bin/mpicc
CFLAGS=-O3 -std=c99
LIBS=

pingpong: pingpong.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm pingpong

