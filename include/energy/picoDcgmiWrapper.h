#pragma once

#include <common/dcgmiLogger.h>

// Init profile thread for DCGMI mesurements
#define PICODCGMI_START( FN , RK ) 						\
    std::thread threadStart;							\
    dcgmiLogger dcgmi_logger ( FN , RK);                                    	\
    if ( RK == 0 ) {								\
        printf("DCGM class created\n");						\
	/* threadStart starts running */					\
        threadStart = std::thread( &dcgmiLogger::getStats, &dcgmi_logger );	\
	/* Neaded for waiting the thread start */				\
        sleep(10);								\
    }

#define PICODCGMI_STOP( RK )								\
    if ( RK == 0 ) {									\
            std::thread threadKill( &dcgmiLogger::killThread, &dcgmi_logger);		\
            threadStart.join( );							\
            threadKill.join( );								\
            printf("Rank %d: thread killed\n", RK );					\
    }
