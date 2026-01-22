
#pragma once

#ifdef USE_NCCL
#include <nccl.h>
#elif defined(USE_RCCL)
#include <rccl/rccl.h>
#elif defined(USE_OCCL)
#include "oneapi/ccl.hpp"
#endif

// Data type mapping for Communicators and Collective data types. 
// NCCL and RCCL have the same types for the communicator and data types of the collectives. Differently, oneCCL uses its own types for the communicator
// and the native C++ types for the collective data types.

namespace common {
    namespace utils {
        namespace data_types {

            // Determine which communicator type to use via default template parameter
            template <typename T = 
            #if defined(USE_ONECCL)
                ccl::communicator
            #elif defined(USE_NCCL)
                ncclComm_t
            #elif defined(USE_RCCL)
                ncclComm_t
            #else
                void   // fallback type if none is defined
            #endif
            >
            struct CommWrapper {
                using type = T;
            };

            template <typename DataType>
            struct DataTypeWrapper {};
            // Specializations
            #ifdef USE_NCCL
            template <>
            struct DataTypeWrapper<float> {
                using type = ncclFloat;
            };
            #elif defined(USE_RCCL)
            template <>
            struct DataTypeWrapper<float> { // RCCL uses the same data type of NCCL
                using type = ncclFloat;
            };
            #elif defined(USE_ONECCL)
            template <>
            struct DataTypeWrapper<float> {
                using type = float;
            };
            #endif

        } // end namespace data_types
    } // end namespace utils
} // end namespace common