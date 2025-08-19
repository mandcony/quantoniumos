#ifndef RFT_CRYPTO_H
#define RFT_CRYPTO_H

#include <vector>
#include <string>

namespace Quantonium {

std::vector<unsigned char> enhanced_rft_encrypt(const std::vector<unsigned char>& plaintext, const std::vector<unsigned char>& key);
std::vector<unsigned char> enhanced_rft_decrypt(const std::vector<unsigned char>& ciphertext, const std::vector<unsigned char>& key);

} // namespace Quantonium

#endif // RFT_CRYPTO_H
