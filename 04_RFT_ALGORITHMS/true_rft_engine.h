#ifndef TRUE_RFT_ENGINE_H
#define TRUE_RFT_ENGINE_H

#include <vector>
#include <complex>
#include <cstdint>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for C interface (only define if not already defined)
#ifndef RFT_RESULT_DEFINED
#define RFT_RESULT_DEFINED
typedef struct {
    float* bins;
    int bin_count;
    float hr;
} RFTResult;
#endif

#ifndef SA_VECTOR_DEFINED
#define SA_VECTOR_DEFINED
typedef struct {
    float* values;
    int count;
} SAVector;
#endif

// C interface functions
int engine_init(void);
void engine_final(void);
RFTResult* rft_run(const char* data, int length);
void rft_free(RFTResult* result);
SAVector* sa_compute(const char* data, int length);
void sa_free(SAVector* vector);
const char* wave_hash(const char* data, int length);
int generate_entropy(uint8_t* buffer, int length);

#ifdef __cplusplus
}

// C++ forward declaration only - actual class definition is in the implementation
class TrueRFTEngine;

#endif // __cplusplus

#endif // TRUE_RFT_ENGINE_H
