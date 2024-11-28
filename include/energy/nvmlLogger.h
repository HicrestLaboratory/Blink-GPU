#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include <nvml.h>
#include <thread>

#define POWERCHUNKSSIZE 1000

#define NVML_CHECK( R ) { if ( R != 0) fprintf(stderr, "Error at line %d: %d\n", __LINE__, R); }


class myPowerSampling {
        public:

        int dev;
        int flag;
        FILE *fpout;
        int nsamples;
        nvmlDevice_t device;
        nvmlReturn_t result;
        unsigned int *power_vec;

        myPowerSampling(int my_dev, char* filename) {
                dev = my_dev;
                nsamples = POWERCHUNKSSIZE ;
                power_vec = (unsigned int*)malloc(sizeof(unsigned int)*nsamples);

                result = nvmlDeviceGetHandleByIndex_v2 ( my_dev, &device );
                NVML_CHECK( result )

                fpout = fopen(filename, "w");
        }

        ~myPowerSampling() {

                printf("Power valuses by %d:\n", dev);
                for (int i=0; i<nsamples; i++) {
                        //printf("\t%i: %llu\n", dev, power_vec[i]);
                        fprintf(fpout, "%u\n", power_vec[i]);
                }
                fclose(fpout);
        }

        void executePowerSampling() {

                printf("Process %d launched %s (line %d)\n", dev, __func__, __LINE__);

                flag = 1;
                int i = 0;
                while (flag) {
                        if (i >= POWERCHUNKSSIZE ) {
                                printf("Process %d realloced the buffer (line %d)\n", dev, __LINE__);
                                nsamples += POWERCHUNKSSIZE ;
                                power_vec = (unsigned int*)realloc(power_vec, sizeof(unsigned int)*nsamples);
                                i = 0;
                        }

                        result = nvmlDeviceGetPowerUsage ( device, &(power_vec[i]) );
                        NVML_CHECK( result )

                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        i++;
                }

                if ( i < POWERCHUNKSSIZE )
                        nsamples -= (POWERCHUNKSSIZE - i);

                printf("Process %d stopped sampling (line %d): %d samples kept\n", dev, __LINE__, nsamples);
        }

        void killThread() {
                // Retrieve a few empty samples
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                // Set loop to false to exit while loop
                flag = 0;
        }
};
