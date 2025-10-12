#ifndef QUANTONIUMOS_SHA256_H
#define QUANTONIUMOS_SHA256_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void sha256_hash(const uint8_t *data, size_t len, uint8_t *output);

#ifdef __cplusplus
}
#endif

#endif /* QUANTONIUMOS_SHA256_H */
