#pragma once

#include <stdlib.h>
#include <inttypes.h>

static uint32_t rng_state = 123456789;

static inline uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

void rand_int8_array(int8_t* array, size_t nelements){
    size_t nextractions = (nelements + 7) / 8 , idx=0;
    for (int i=0; i<nextractions; i++) {
        uint32_t r = xorshift32();
        for (int j = 0; j < 8; j++) {
            int val = (r & 0xF) - 8;  // 4 bits → [-8, 7]
            r >>= 4;
            if(idx<nelements) array[idx] = val;
            idx++;
        }
    }
}
