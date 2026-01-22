
#pragma once

#ifdef USE_NCCL
#include <nccl.h>
#elif defined(USE_RCCL)
#include <rccl/rccl.h>
#elif defined(USE_OCCL)
#include "oneapi/ccl.hpp"
#endif

namespace common {
    namespace utils {

        template <typename CommType>
        struct CommWrapper {};

        // Specializations
        #ifdef USE_NCCL
        template <>
        struct CommWrapper<ncclComm_t> {
            using type = ncclComm_t;
        };
        #elif defined(USE_RCCL)
        template <>
        struct CommWrapper<rcclComm_t> {
            using type = rcclComm_t;
        };
        #elif defined(USE_ONECCL)
        template <>
        struct CommWrapper<ccl::communicator> {
            using type = ccl::communicator;
        };
        #endif
    } // end namespace utils
} // end namespace common