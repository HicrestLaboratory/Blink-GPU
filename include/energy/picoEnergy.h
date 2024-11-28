#pragma once

#define ENERGY_FILENAME_LENGHT 200
#define ENERGY_PATH "energy/results/"

#ifdef PICODCGMI
#define ENERGY_FILENAME_PREFIX "dcgmiMesures"
#else
#define ENERGY_FILENAME_PREFIX "nvmlMesures"
#endif

#define PICOENERGY_FILENAME_VAR energymesures_filename

#define PICOENERGY_DEFINE_FILENAME( BS, LC )                                            \
    char PICOENERGY_FILENAME_VAR[ENERGY_FILENAME_LENGHT];                                \
    sprintf(PICOENERGY_FILENAME_VAR, "%s%s_%s_%s_%d_%d.csv",                             \
            ENERGY_PATH, ENERGY_FILENAME_PREFIX, MYBENCH_CODE, MYIMPL_CODE, BS, LC);

void energyCompileTimeCheck(void) {
#if !defined(PICODCGMI) && !defined(PICONVML)
	fprintf(stderr, "Error: error in compiling, ENERGY flag must metch with a PICODCGMI or PICONVML define\n");
	exit(__LINE__);
#endif

#if defined(PICODCGMI) && defined(PICONVML)
	fprintf(stderr, "Error: only one between PICODCGMI or PICONVML can be defined\n");
	exit(__LINE__);
#endif
	return;
}


#ifdef PICODCGMI
/* ------------------------------------------------------------------------------
 * 									DCGMI
 * ------------------------------------------------------------------------------
 */
#include <energy/dcgmiLogger.h>

// Init profile thread for DCGMI mesurements
#define PICODCGMI_START( BS, LC, RK )                                           \
    PICOENERGY_DEFINE_FILENAME( BS, LC )                                        \
    std::thread threadStart;                                                    \
    dcgmiLogger dcgmi_logger ( PICOENERGY_FILENAME_VAR , RK);                   \
    if ( RK == 0 ) {                                                            \
        printf("DCGM class created\n");                                         \
	/* threadStart starts running */                                            \
        threadStart = std::thread( &dcgmiLogger::getStats, &dcgmi_logger );     \
	/* Neaded for waiting the thread start */                                   \
        sleep(10);                                                              \
    }

#define PICODCGMI_STOP( RK )                                                    \
    if ( RK == 0 ) {                                                            \
            std::thread threadKill( &dcgmiLogger::killThread, &dcgmi_logger);   \
            threadStart.join( );                                                \
            threadKill.join( );                                                 \
            printf("Rank %d: thread killed\n", RK );                            \
    }

#else
/* ------------------------------------------------------------------------------
 * 									NVML
 * ------------------------------------------------------------------------------
 */

#include <nvml.h>

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

#define PICONVML_ENERGYCOUNTER_NAME piconvml_energy
#define PICONVML_ENERGYCOUNTER_START CONCAT(PICONVML_ENERGYCOUNTER_NAME, _start)
#define PICONVML_ENERGYCOUNTER_STOP  CONCAT(PICONVML_ENERGYCOUNTER_NAME, _stop)

#define PICONVML_DEFINE                                     \
		nvmlInit();											\
		energyCompileTimeCheck();							\
        unsigned long long PICONVML_ENERGYCOUNTER_START;	\
        unsigned long long PICONVML_ENERGYCOUNTER_STOP;

#define PICONVML_ENERGY_START( BS, LC, DV )						                              \
		PICOENERGY_DEFINE_FILENAME( BS, LC )					                              \
		picoNvmlTotalEnergy(DV, &PICONVML_ENERGYCOUNTER_START);                               \
		std::thread threadStart;                                                              \
		myPowerSampling power_samples ( RK , PICOENERGY_FILENAME_VAR );                       \
		threadStart = std::thread( &myPowerSampling::executePowerSampling, &power_samples );  \
		sleep(10);


#define PICONVML_ENERGY PICONVML_ENERGYCOUNTER_STOP - PICONVML_ENERGYCOUNTER_START

#define PICONVML_ENERGY_STOP( DV )  							                \
        std::thread threadKill( &myPowerSampling::killThread, &power_samples);  \
        threadStart.join( );                                                    \
        threadKill.join( );                                                     \
		picoNvmlTotalEnergy(DV, &PICONVML_ENERGYCOUNTER_STOP);	                \
		MPI_Barrier(MPI_COMM_WORLD);							                \
		if (rank == 0) printf("Delta power for device:\n");		                \
		printf("\t%i: %u\n", DV, PICONVML_ENERGY);

void picoNvmlTotalEnergy(int my_dev, unsigned long long* energy) {

        nvmlDevice_t device;
        nvmlReturn_t result;
        nvmlDeviceGetHandleByIndex_v2 ( my_dev, &device );
        result = nvmlDeviceGetTotalEnergyConsumption(device, energy);
        if ( result != 0) fprintf(stderr, "Error at line %d: %d\n", __LINE__, result);

}

void picoNvmlInstantPower(int my_dev, unsigned int* power) {

        nvmlDevice_t device;
        nvmlReturn_t result;
        nvmlDeviceGetHandleByIndex_v2 ( my_dev, &device );
        result = nvmlDeviceGetPowerUsage ( device, power );
        if ( result != 0) fprintf(stderr, "Error at line %d: %d\n", __LINE__, result);

}

#endif

/* ------------------------------------------------------------------------------
 * 									Common
 * ------------------------------------------------------------------------------
 */

#ifdef PICODCGMI
#define PICOENERGY_DEFINE energyCompileTimeCheck();
#define PICOENERGY_START( BS, LC, RK ) PICODCGMI_START( BS , LC , RK )
#define PICOENERGY_STOP( RK ) PICODCGMI_STOP(RK)
#else
#define PICOENERGY_DEFINE PICONVML_DEFINE
#define PICOENERGY_START( BS, LC, RK ) PICONVML_ENERGY_START( BS , LC , RK )
#define PICOENERGY_STOP( RK ) PICONVML_ENERGY_STOP(RK)
#endif
