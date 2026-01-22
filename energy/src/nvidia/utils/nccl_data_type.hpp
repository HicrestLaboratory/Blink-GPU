#include <nccl.h>
#include <cstdint>
template <typename T>
struct nccl_type_traits;

template <>
struct nccl_type_traits<float> {
    static constexpr ncclDataType_t type = ncclFloat;
};

template <>
struct nccl_type_traits<double> {
    static constexpr ncclDataType_t type = ncclDouble;
};

template <>
struct nccl_type_traits<int> {
    static constexpr ncclDataType_t type = ncclInt32;
};

template <>
struct nccl_type_traits<int64_t> {
    static constexpr ncclDataType_t type = ncclInt64;
};

template <>
struct nccl_type_traits<uint8_t> {
    static constexpr ncclDataType_t type = ncclUint8;
};
