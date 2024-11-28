#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include <nvml.h>
#include <thread>

#define POWERCHUNKSSIZE 1000

#define NVML_CHECK( R ) { if ( R != 0) fprintf(stderr, "NVML error at line %d of file %s: %d\n", __LINE__, __FILE__, R); }


class myPowerSampling {
        public:

        int dev;
        int flag;
        FILE *fpout;
        int nsamples;
        int nrealloc;
        nvmlDevice_t device;
        nvmlReturn_t result;
	unsigned int *temperature_vec;
        unsigned int *power_vec;

        myPowerSampling(int my_dev, char* filename) {
                dev = my_dev;
                nrealloc = 0;
                nsamples = POWERCHUNKSSIZE ;
                power_vec = (unsigned int*)malloc(sizeof(unsigned int)*nsamples);
		temperature_vec = (unsigned int*)malloc(sizeof(unsigned int)*nsamples);

                result = nvmlDeviceGetHandleByIndex_v2 ( my_dev, &device );
                NVML_CHECK( result )

                fpout = fopen(filename, "w");
		fprintf(fpout, "#deviceId,InstantPower,InstantTemperature\n");
        }

        ~myPowerSampling() {

                for (int i=0; i<nsamples; i++)
                        fprintf(fpout, "%d,%u,%u\n", dev, power_vec[i], temperature_vec[i]);
                fclose(fpout);
		printf("Device %d printed its sampling results on file\n", dev);
        }

        void executePowerSampling() {

                printf("Process %d launched %s (line %d)\n", dev, __func__, __LINE__);

                flag = 1;
                int i = 0;
                while (flag) {
                        if (i >= POWERCHUNKSSIZE ) {
//                                 printf("Process %d realloced the buffer (line %d)\n", dev, __LINE__);
                                nrealloc += 1;
                                nsamples += POWERCHUNKSSIZE ;
                                power_vec = (unsigned int*)realloc(power_vec, sizeof(unsigned int)*nsamples);
				temperature_vec = (unsigned int*)realloc(temperature_vec, sizeof(unsigned int)*nsamples);
                                i = 0;
                        }

                        result = nvmlDeviceGetPowerUsage ( device, &(power_vec[i]) );
                        NVML_CHECK( result )
			result = nvmlDeviceGetTemperature ( device, NVML_TEMPERATURE_GPU, &(temperature_vec[i]) );
			NVML_CHECK( result )

                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        i++;
                }

                if ( i < POWERCHUNKSSIZE )
                        nsamples -= (POWERCHUNKSSIZE - i);

                printf("Process %d stopped sampling (line %d): %d reallocations, %d samples kept\n", dev, __LINE__, nrealloc, nsamples);
        }

        void killThread() {
                // Retrieve a few empty samples
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                // Set loop to false to exit while loop
                flag = 0;
        }
};
