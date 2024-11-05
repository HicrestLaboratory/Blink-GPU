#pragma once

#include <energy/dcgmiLogger.h>

#define ENERGY_FILENAME_PREFIX "dcgmiMesures"
#define ENERGY_FILENAME_LENGHT 200
#define ENERGY_PATH "energy/results/"

#define PICODCGMI_DEFINE_FILENAME( BS, LC )                                             \
    char energymesures_filename[ENERGY_FILENAME_LENGHT];                                \
    sprintf(energymesures_filename, "%s%s_%s_%s_%d_%d.csv",                             \
            ENERGY_PATH, ENERGY_FILENAME_PREFIX, MYBENCH_CODE, MYIMPL_CODE, BS, LC);

// Init profile thread for DCGMI mesurements
#define PICODCGMI_START( BS, LC, RK )                                           \
    PICODCGMI_DEFINE_FILENAME( BS, LC )                                         \
    std::thread threadStart;                                                    \
    dcgmiLogger dcgmi_logger ( energymesures_filename , RK);                    \
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
