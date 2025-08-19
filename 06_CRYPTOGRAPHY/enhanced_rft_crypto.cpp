#include <vector>
#include <array>
#include <cstring>
#include <cstdint>
#include <memory>
#include <stdexcept>

// Optimized Enhanced RFT Crypto Implementation
class EnhancedRFTCrypto {
private:
    // AES S-box for nonlinearity
    static constexpr std::array<uint8_t, 256> SBOX = {
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    };

    // Optimized permutation table for MDS-like mixing
    static constexpr std::array<uint8_t, 256> PERM_TABLE = {
        7, 12, 1, 14, 9, 2, 15, 8, 3, 10, 5, 0, 11, 4, 13, 6,
        2, 9, 6, 3, 12, 15, 8, 1, 10, 7, 4, 13, 0, 11, 14, 5,
        11, 4, 13, 6, 7, 12, 1, 14, 9, 2, 15, 8, 3, 10, 5, 0,
        8, 15, 2, 9, 6, 3, 12, 5, 0, 11, 14, 7, 4, 13, 10, 1,
        13, 6, 11, 4, 1, 14, 7, 12, 3, 8, 15, 2, 9, 0, 5, 10,
        4, 11, 14, 1, 8, 5, 2, 15, 12, 3, 6, 9, 10, 7, 0, 13,
        15, 2, 9, 12, 5, 8, 3, 6, 1, 14, 11, 4, 7, 10, 13, 0,
        6, 13, 0, 7, 10, 1, 14, 11, 4, 9, 2, 15, 12, 5, 8, 3,
        1, 8, 15, 2, 11, 4, 9, 6, 13, 0, 7, 10, 5, 12, 3, 14,
        10, 7, 4, 13, 0, 11, 6, 9, 2, 15, 12, 5, 8, 1, 14, 3,
        5, 0, 7, 10, 13, 6, 11, 4, 15, 2, 9, 12, 1, 14, 7, 8,
        12, 5, 2, 15, 8, 1, 14, 7, 10, 13, 0, 3, 6, 9, 4, 11,
        9, 14, 3, 0, 7, 10, 13, 2, 5, 12, 1, 6, 11, 8, 15, 4,
        0, 3, 10, 5, 14, 9, 4, 13, 6, 1, 8, 11, 2, 15, 12, 7,
        3, 10, 5, 8, 15, 2, 1, 0, 7, 4, 13, 14, 11, 6, 9, 12,
        14, 1, 8, 11, 4, 7, 10, 3, 0, 5, 2, 9, 12, 3, 6, 15
    };

    // Cached key schedule
    mutable std::vector<uint8_t> last_key;
    mutable std::vector<uint64_t> cached_key_schedule;
    mutable std::vector<int> cached_rot_a, cached_rot_b, cached_rot_c;
    mutable std::vector<uint64_t> cached_rc32;

public:
    // Fast F-function with optimized ARX structure
    // GF(2^8) multiplication helpers for enhanced diffusion
    static inline uint8_t xtime(uint8_t x) {
        return (x << 1) ^ (((x >> 7) & 1) * 0x1b);
    }
    static inline uint8_t mul2(uint8_t x) { return xtime(x); }
    static inline uint8_t mul3(uint8_t x) { return xtime(x) ^ x; }

    // Enhanced MixColumns for stronger linear diffusion
    static inline void mixcolumns_4x4(uint8_t s[16]) {
        for (int i = 0; i < 4; i++) {
            uint8_t* col = s + (i * 4);
            uint8_t a = col[0], b = col[1], c = col[2], d = col[3];
            col[0] = mul2(a) ^ mul3(b) ^ c ^ d;
            col[1] = a ^ mul2(b) ^ mul3(c) ^ d;
            col[2] = a ^ b ^ mul2(c) ^ mul3(d);
            col[3] = mul3(a) ^ b ^ c ^ mul2(d);
        }
    }

    uint64_t F(uint64_t x, uint64_t k) const {
        alignas(16) uint8_t state[16];
        // Initialize state with input and key material
        std::memcpy(state, &x, 8);
        std::memcpy(state + 8, &k, 8);

        // Enhanced ARX with 4 quarter-rounds using 3 independent rotates
        for (int qr = 0; qr < 4; qr++) {
            uint64_t* q = reinterpret_cast<uint64_t*>(state + (qr % 2) * 8);
            // Use round-dependent rotation amounts from cached schedule
            int rot_a = cached_rot_a[qr % 48];
            int rot_b = cached_rot_b[qr % 48];
            int rot_c = cached_rot_c[qr % 48];
            uint64_t rc = cached_rc32[qr % 48];
            *q += rc;
            *q = ((*q << rot_a) | (*q >> (64 - rot_a)));
            *q ^= k;
            *q = ((*q << rot_b) | (*q >> (64 - rot_b)));
            *q += k;
            *q = ((*q << rot_c) | (*q >> (64 - rot_c)));
        }

        // AES S-box substitution on all 16 bytes
        for (int i = 0; i < 16; i++) {
            state[i] = SBOX[state[i]];
        }

        // MixColumns linear diffusion over all 16 bytes
        mixcolumns_4x4(state);

        // ShiftRows-like row rotation for alignment breaking
        uint8_t temp = state[1];
        state[1] = state[5]; state[5] = state[9]; state[9] = state[13]; state[13] = temp;
        temp = state[2];
        state[2] = state[10]; state[10] = temp;
        temp = state[6];
        state[6] = state[14]; state[14] = temp;
        temp = state[3];
        state[3] = state[15]; state[15] = state[11]; state[11] = state[7]; state[7] = temp;

        // Dual key injection: XOR + additive
        uint64_t* result_ptr = reinterpret_cast<uint64_t*>(state);
        result_ptr[0] ^= k;
        result_ptr[1] += k;
        return result_ptr[0] ^ result_ptr[1];
    }

    // Optimized key schedule with caching and enhanced parameters
    void generate_key_schedule(const std::vector<uint8_t>& key) const {
        // Check if we can reuse cached key schedule
        if (key == last_key && !cached_key_schedule.empty()) {
            return;
        }
        last_key = key;
        cached_key_schedule.clear();
        cached_key_schedule.reserve(48);

        // Initialize rotation parameter caches
        cached_rot_a.clear(); cached_rot_a.reserve(48);
        cached_rot_b.clear(); cached_rot_b.reserve(48);
        cached_rot_c.clear(); cached_rot_c.reserve(48);
        cached_rc32.clear(); cached_rc32.reserve(48);

        // Convert key to 64-bit words for efficient processing
        std::vector<uint64_t> key_words((key.size() + 7) / 8, 0);
        for (size_t i = 0; i < key.size(); i++) {
            key_words[i / 8] |= static_cast<uint64_t>(key[i]) << ((i % 8) * 8);
        }

        // Generate round keys and rotation parameters
        for (int round = 0; round < 48; round++) {
            uint64_t round_key = 0;
            // Combine multiple key words with round-dependent operations
            for (size_t i = 0; i < key_words.size(); i++) {
                uint64_t word = key_words[i];
                word = ((word << (round % 64)) | (word >> (64 - (round % 64))));
                word ^= static_cast<uint64_t>(round) * 0x517CC1B727220A95ULL;
                round_key ^= word;
            }
            // Additional mixing with F-function
            if (round > 0) {
                // Use simple ARX for key schedule to avoid circular dependency
                round_key = (round_key + cached_key_schedule[round - 1]) & 0xFFFFFFFFFFFFFFFFULL;
                round_key = ((round_key << 31) | (round_key >> 33));
                round_key ^= 0x6A09E667F3BCC908ULL;
            }
            cached_key_schedule.push_back(round_key);

            // Generate three independent rotation parameters per round
            cached_rot_a.push_back(7 + (round % 26));   // 7-32
            cached_rot_b.push_back(13 + (round % 19));  // 13-31
            cached_rot_c.push_back(17 + (round % 16));  // 17-32

            // Round constants for enhanced key-driven degrees of freedom
            cached_rc32.push_back(round_key ^ (static_cast<uint64_t>(round) * 0x9E3779B97F4A7C15ULL));
        }
    }

    // High-performance encryption with optimized memory layout
    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext, const std::vector<uint8_t>& key) const {
        generate_key_schedule(key);

        // Always work with complete 16-byte blocks
        size_t block_count = (plaintext.size() + 15) / 16;
        std::vector<uint8_t> result(block_count * 16);

        // Process each 16-byte block
        for (size_t block = 0; block < block_count; block++) {
            size_t offset = block * 16;
            // Load block with padding if needed
            alignas(16) uint64_t left = 0, right = 0;
            size_t bytes_to_copy = std::min(size_t(16), plaintext.size() - offset);
            if (bytes_to_copy >= 8) {
                std::memcpy(&left, &plaintext[offset], 8);
                if (bytes_to_copy > 8) {
                    std::memcpy(&right, &plaintext[offset + 8], bytes_to_copy - 8);
                }
            } else {
                std::memcpy(&left, &plaintext[offset], bytes_to_copy);
            }

            // Pre-whitening with block position for enhanced message diffusion
            left ^= static_cast<uint64_t>(block) * 0x517CC1B727220A95ULL;
            right ^= static_cast<uint64_t>(block) * 0x9E3779B97F4A7C15ULL;

            // Cross-block mixing for better avalanche
            if (block > 0) {
                // Mix with previous block's data for inter-block diffusion
                size_t prev_offset = (block - 1) * 16;
                uint64_t prev_left = 0, prev_right = 0;
                if (prev_offset < plaintext.size()) {
                    size_t prev_bytes = std::min(size_t(8), plaintext.size() - prev_offset);
                    std::memcpy(&prev_left, &plaintext[prev_offset], prev_bytes);
                    if (plaintext.size() > prev_offset + 8) {
                        size_t right_bytes = std::min(size_t(8), plaintext.size() - prev_offset - 8);
                        std::memcpy(&prev_right, &plaintext[prev_offset + 8], right_bytes);
                    }
                }
                left ^= prev_left;
                right ^= prev_right;
            }

            // 48-round Feistel with proper swapping
            for (int round = 0; round < 48; round++) {
                uint64_t new_left = right;
                uint64_t new_right = left ^ F(right, cached_key_schedule[round]);
                left = new_left;
                right = new_right;
            }

            // Store result
            std::memcpy(&result[offset], &left, 8);
            std::memcpy(&result[offset + 8], &right, 8);
        }
        return result;
    }

    // High-performance decryption with proper Feistel reversal
    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext, const std::vector<uint8_t>& key) const {
        generate_key_schedule(key);

        // Ciphertext should be multiple of 16 bytes (as produced by encrypt)
        if (ciphertext.size() % 16 != 0) {
            throw std::invalid_argument("Ciphertext size must be multiple of 16 bytes");
        }
        size_t block_count = ciphertext.size() / 16;
        std::vector<uint8_t> result(ciphertext.size());

        // Process each 16-byte block (reverse order for cross-block dependencies)
        for (int block = block_count - 1; block >= 0; block--) {
            size_t offset = block * 16;
            // Load block
            alignas(16) uint64_t left, right;
            std::memcpy(&left, &ciphertext[offset], 8);
            std::memcpy(&right, &ciphertext[offset + 8], 8);

            // 48-round Feistel in reverse with proper swapping
            for (int round = 47; round >= 0; round--) {
                uint64_t new_right = left;
                uint64_t new_left = right ^ F(left, cached_key_schedule[round]);
                left = new_left;
                right = new_right;
            }

            // Reverse pre-whitening
            left ^= static_cast<uint64_t>(block) * 0x517CC1B727220A95ULL;
            right ^= static_cast<uint64_t>(block) * 0x9E3779B97F4A7C15ULL;

            // Store result first, then apply cross-block reversal
            std::memcpy(&result[offset], &left, 8);
            std::memcpy(&result[offset + 8], &right, 8);
        }

        // Apply cross-block diffusion reversal in forward order
        for (size_t block = 1; block < block_count; block++) {
            size_t offset = block * 16;
            size_t prev_offset = (block - 1) * 16;
            uint64_t left, right, prev_left, prev_right;
            std::memcpy(&left, &result[offset], 8);
            std::memcpy(&right, &result[offset + 8], 8);
            std::memcpy(&prev_left, &result[prev_offset], 8);
            std::memcpy(&prev_right, &result[prev_offset + 8], 8);
            left ^= prev_left;
            right ^= prev_right;
            std::memcpy(&result[offset], &left, 8);
            std::memcpy(&result[offset + 8], &right, 8);
        }
        return result;
    }

    // Batch encryption for improved throughput
    std::vector<std::vector<uint8_t>> encrypt_batch(const std::vector<std::vector<uint8_t>>& plaintexts, const std::vector<uint8_t>& key) const {
        generate_key_schedule(key);
        std::vector<std::vector<uint8_t>> results;
        results.reserve(plaintexts.size());
        for (const auto& plaintext : plaintexts) {
            results.push_back(encrypt(plaintext, key));
        }
        return results;
    }
}; 