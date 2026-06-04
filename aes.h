#ifndef AES_H
#define AES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define AES_BLOCK_SIZE 16
#define AES_128_KEY_SIZE 16
#define AES_192_KEY_SIZE 24
#define AES_256_KEY_SIZE 32
#define AES_128_ROUNDS 10
#define AES_192_ROUNDS 12
#define AES_256_ROUNDS 14
#define AES_MAX_ROUNDS AES_256_ROUNDS
#define AES_MAX_EXPANDED_KEY_SIZE ((AES_MAX_ROUNDS + 1) * AES_BLOCK_SIZE)

typedef struct {
    size_t key_size;
    int key_words;
    int rounds;
    uint8_t round_keys[AES_MAX_EXPANDED_KEY_SIZE];
} AESContext;

int aes_init(AESContext *ctx, const uint8_t *key, size_t key_size);

void aes_encrypt_ecb_normal(const uint8_t *input,
                            uint8_t *output,
                            size_t size,
                            const AESContext *ctx);

void aes_encrypt_ecb_trace(const uint8_t *input,
                           uint8_t *output,
                           size_t size,
                           const AESContext *ctx,
                           FILE *trace);

void aes_decrypt_ecb_normal(const uint8_t *input,
                            uint8_t *output,
                            size_t size,
                            const AESContext *ctx);

void aes_decrypt_ecb_trace(const uint8_t *input,
                           uint8_t *output,
                           size_t size,
                           const AESContext *ctx,
                           FILE *trace);

#endif
