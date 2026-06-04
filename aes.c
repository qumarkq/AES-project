#include "aes.h"

#include <string.h>
// #include <emmintrin.h>
#include <tmmintrin.h>
#include <stdlib.h>
#include <omp.h>

/* AES lookup tables */
static const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t inv_sbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

static const uint8_t rcon[11] = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10,
    0x20, 0x40, 0x80, 0x1b, 0x36
};

static void add_round_key(uint8_t *state,
                          const uint8_t *round_key);
static void sub_bytes(uint8_t *state);
static void shift_rows(uint8_t *state);
static inline __m128i xtime_vec(__m128i x);
static inline __m128i rotate_column_bytes_left(__m128i x);
static void mix_columns(uint8_t *state);
static void inv_sub_bytes(uint8_t *state);
static void inv_shift_rows(uint8_t *state);
static void inv_mix_columns(uint8_t *state);


typedef struct {
    char *data;
    size_t size;
    size_t capacity;
} TraceBuffer;

static int append_trace_state(TraceBuffer *trace,
                              const char *label,
                              size_t block_index,
                              int round,
                              const uint8_t state[AES_BLOCK_SIZE]);

static int aes_encrypt_block_with_trace(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx,
    TraceBuffer *trace,
    size_t block_index);

static int aes_decrypt_block_with_trace(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx,
    TraceBuffer *trace,
    size_t block_index);


static inline __m128i xtime_vec(__m128i x) {
    /*
     * 0x80 用來檢查每個 byte 的最高 bit。
     * 如果最高 bit 是 1，左移後會超出 8 bits，
     * 需要再 XOR 0x1b 做 reduction。
     */
    const __m128i msb_mask = _mm_set1_epi8((char)0x80);

    /*
     * AES GF(2^8) 裡，乘以 2 時若發生溢位，
     * 需要 XOR 0x1b。
     */
    const __m128i reduction = _mm_set1_epi8(0x1b);

    /*
     * 保留每個 byte 的最高 bit。
     * high_bits 的每個 byte 只可能是 0x80 或 0x00。
     */
    __m128i high_bits = _mm_and_si128(x, msb_mask);

    /*
     * shifted = x << 1。
     * _mm_add_epi8(x, x) 等價於每個 byte 乘以 2。
     *
     * 這裡使用加法是因為 SSE2 沒有直接的 8-bit left shift。
     * 對 byte 來說，x + x 的低 8 bits 等價於 x 左移 1 bit。
     */
    __m128i shifted = _mm_add_epi8(x, x);

    /*
     * 檢查哪些 byte 原本最高 bit 是 1。
     * 若 high_bits == 0x80，該 byte 會變成 0xff；
     * 否則會變成 0x00。
     *
     * 這會產生一個 mask：
     *   0xff 表示該 byte 需要 XOR 0x1b
     *   0x00 表示不需要 reduction
     */
    __m128i needs_reduction = _mm_cmpeq_epi8(high_bits, msb_mask);

    /*
     * 根據 mask 選出需要套用的 reduction。
     *
     * 如果 needs_reduction 是 0xff：
     *   0xff & 0x1b = 0x1b
     *
     * 如果 needs_reduction 是 0x00：
     *   0x00 & 0x1b = 0x00
     */
    __m128i reduced = _mm_and_si128(needs_reduction, reduction);

    /*
     * 如果原本最高 bit 是 1：
     *   回傳 shifted ^ 0x1b
     *
     * 如果原本最高 bit 是 0：
     *   回傳 shifted ^ 0x00，也就是 shifted
     */
    return _mm_xor_si128(shifted, reduced);
}

static inline __m128i rotate_column_bytes_left(__m128i x) {
    /*
     * 每個 32-bit lane 右移 8 bits。
     * 從記憶體 byte 順序來看：
     *   [s0, s1, s2, s3] -> [s1, s2, s3, 00]
     */
    __m128i right = _mm_srli_epi32(x, 8);

    /*
     * 每個 32-bit lane 左移 24 bits。
     * 從記憶體 byte 順序來看：
     *   [s0, s1, s2, s3] -> [00, 00, 00, s0]
     */
    __m128i left = _mm_slli_epi32(x, 24);

    /*
     * 合併兩個位移結果，完成每個 column 的循環左轉。
     */
    return _mm_or_si128(right, left);
}


/*
 * AES AddRoundKey。
 *
 * AddRoundKey 的操作是：
 *   state = state ^ round_key
 *
 * AES state 和每一輪 round key 都是 16 bytes。
 * 這裡使用 SIMD 一次載入 16 bytes，並用一個 XOR 指令完成整個 block。
 */
static void add_round_key(uint8_t *state,
                          const uint8_t *round_key) {
    __m128i state_vec;
    __m128i key_vec;
    __m128i result_vec;

    /*
     * 載入目前的 16-byte AES state。
     * 使用 loadu，所以 state 不需要 16-byte 對齊。
     */
    state_vec = _mm_loadu_si128((const __m128i *)state);

    /*
     * 載入目前 round 的 16-byte round key。
     * 不管 AES-128 / AES-192 / AES-256，
     * 每一輪實際 XOR 的 round key 都是 16 bytes。
     */
    key_vec = _mm_loadu_si128((const __m128i *)round_key);

    /*
     * 對 16 bytes 同時做 XOR：
     *   result_vec[i] = state_vec[i] ^ key_vec[i]
     */
    result_vec = _mm_xor_si128(state_vec, key_vec);

    /*
     * 將 AddRoundKey 後的結果寫回 state。
     */
    _mm_storeu_si128((__m128i *)state, result_vec);
}

/* Byte substitution */
static void sub_bytes(uint8_t *state) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        state[i] = sbox[state[i]];
    }
}

static void inv_sub_bytes(uint8_t *state) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        state[i] = inv_sbox[state[i]];
    }
}

/*
 * AES state is stored in column-major order:
 *   0  4  8  12
 *   1  5  9  13
 *   2  6  10 14
 *   3  7  11 15
 */
/* Row shifting */
static void shift_rows(uint8_t *state) {
    // uint8_t temp;


    /* Row 1: shift left by 1 */
    // temp = state[1];
    // state[1] = state[5];
    // state[5] = state[9];
    // state[9] = state[13];
    // state[13] = temp;

    // /* Row 2: shift left by 2 */
    // temp = state[2];
    // state[2] = state[10];
    // state[10] = temp;

    // temp = state[6];
    // state[6] = state[14];
    // state[14] = temp;

    // /* Row 3: shift left by 3 */
    // temp = state[15];
    // state[15] = state[11];
    // state[11] = state[7];
    // state[7] = state[3];
    // state[3] = temp;
    const __m128i mask = _mm_setr_epi8(
            0, 5, 10, 15,
            4, 9, 14, 3,
            8, 13, 2, 7,
            12, 1, 6, 11
    );
    __m128i s = _mm_loadu_si128((const __m128i *)state);
    s = _mm_shuffle_epi8(s, mask);
    _mm_storeu_si128((__m128i *)state, s);

}

static void inv_shift_rows(uint8_t *state) {
    // uint8_t temp;

    // /* Row 1: shift right by 1 */
    // temp = state[13];
    // state[13] = state[9];
    // state[9] = state[5];
    // state[5] = state[1];
    // state[1] = temp;

    // /* Row 2: shift right by 2 */
    // temp = state[2];
    // state[2] = state[10];
    // state[10] = temp;

    // temp = state[6];
    // state[6] = state[14];
    // state[14] = temp;

    // /* Row 3: shift right by 3 */
    // temp = state[3];
    // state[3] = state[7];
    // state[7] = state[11];
    // state[11] = state[15];
    // state[15] = temp;
    const __m128i mask = _mm_setr_epi8(
        0, 13, 10, 7,
        4, 1, 14, 11,
        8, 5, 2, 15,
        12, 9, 6, 3
    );

    __m128i s = _mm_loadu_si128((const __m128i *)state);
    s = _mm_shuffle_epi8(s, mask);
    _mm_storeu_si128((__m128i *)state, s);
}

/*
 * 在 GF(2^8) 上執行 AES MixColumns。
 *
 * AES 的一個 column 為：
 *   [s0, s1, s2, s3]
 *
 * MixColumns 會轉換成：
 *   [2*s0 ^ 3*s1 ^ s2    ^ s3,
 *    s0    ^ 2*s1 ^ 3*s2 ^ s3,
 *    s0    ^ s1    ^ 2*s2 ^ 3*s3,
 *    3*s0 ^ s1    ^ s2    ^ 2*s3]
 *
 * 這個 SIMD 版本使用等價公式：
 *   result = column ^ column_xor ^ xtime(column ^ sl_1_column)
 *
 * 其中：
 *   column_xor = s0 ^ s1 ^ s2 ^ s3
 *   xtime(x)   = AES GF(2^8) 裡的 2*x
 */
static void mix_columns(uint8_t *state) {
    __m128i column;       /* 每個 column: [s0, s1, s2, s3] */
    __m128i sl_1_column;  /* 每個 column 左轉 1 byte: [s1, s2, s3, s0] */
    __m128i sl_2_column;  /* 每個 column 左轉 2 bytes: [s2, s3, s0, s1] */
    __m128i sl_3_column;  /* 每個 column 左轉 3 bytes: [s3, s0, s1, s2] */
    __m128i column_xor;
    __m128i adjacent_xor;
    __m128i doubled_adjacent_xor;
    __m128i result;

    /*
     * 讀入 16-byte AES state。
     * state 在記憶體中是 4 個連續的 4-byte column。
     */
    column = _mm_loadu_si128((const __m128i *)state);

    /*
     * 對每個 4-byte column 分別做循環左轉。
     * 這樣可以把同一個 column 裡的其他 byte 對齊到相同位置，
     * 方便後面用 SIMD 同時計算 4 個 column。
     */
    sl_1_column = rotate_column_bytes_left(column);
    sl_2_column = rotate_column_bytes_left(sl_1_column);
    sl_3_column = rotate_column_bytes_left(sl_2_column);

    /*
     * 計算每個 column 的 XOR 總和：
     *   sum = s0 ^ s1 ^ s2 ^ s3
     *
     * 因為：
     *   column      = [s0, s1, s2, s3]
     *   sl_1_column = [s1, s2, s3, s0]
     *   sl_2_column = [s2, s3, s0, s1]
     *   sl_3_column = [s3, s0, s1, s2]
     *
     * 四個 vector XOR 後，每個 byte 位置都會得到同一個 sum：
     *   column_xor = [sum, sum, sum, sum]
     *
     * SIMD 會同時對 4 個 AES columns 做這件事。
     */
    column_xor = _mm_xor_si128(column, sl_1_column);
    column_xor = _mm_xor_si128(column_xor, sl_2_column);
    column_xor = _mm_xor_si128(column_xor, sl_3_column);

    /*
     * 計算 MixColumns 等價公式中的：
     *   xtime(column ^ sl_1_column)
     *
     * 其中：
     *   column ^ sl_1_column
     *     = [s0^s1, s1^s2, s2^s3, s3^s0]
     *
     * xtime(x) 表示在 AES GF(2^8) 中乘以 2，因此：
     *   doubled_adjacent_xor
     *     = [2*(s0^s1), 2*(s1^s2), 2*(s2^s3), 2*(s3^s0)]
     *
     * 這個項目用來產生 MixColumns 公式裡的 2*s 和 3*s。
     */
    adjacent_xor = _mm_xor_si128(column, sl_1_column);
    doubled_adjacent_xor = xtime_vec(adjacent_xor);

    /*
     * 套用 MixColumns 的等價公式：
     *   result = column ^ column_xor ^ doubled_adjacent_xor
     *
     * 以第一個 byte 為例：
     *   result0 = s0 ^ (s0^s1^s2^s3) ^ 2*(s0^s1)
     *           = 2*s0 ^ 3*s1 ^ s2 ^ s3
     *
     * 這正是 AES MixColumns 第一列的結果。
     */
    result = _mm_xor_si128(column, column_xor);
    result = _mm_xor_si128(result, doubled_adjacent_xor);

    /* 將 MixColumns 後的結果寫回 state。 */
    _mm_storeu_si128((__m128i *)state, result);
}

/*
 * 在 GF(2^8) 上執行 AES InvMixColumns。
 *
 * AES inverse MixColumns 會將一個 column：
 *   [s0, s1, s2, s3]
 *
 * 轉換成：
 *   [0e*s0 ^ 0b*s1 ^ 0d*s2 ^ 09*s3,
 *    09*s0 ^ 0e*s1 ^ 0b*s2 ^ 0d*s3,
 *    0d*s0 ^ 09*s1 ^ 0e*s2 ^ 0b*s3,
 *    0b*s0 ^ 0d*s1 ^ 09*s2 ^ 0e*s3]
 *
 * 這個 SIMD 版本先建立 column 的循環左轉版本：
 *   column      = [s0, s1, s2, s3]
 *   sl_1_column = [s1, s2, s3, s0]
 *   sl_2_column = [s2, s3, s0, s1]
 *   sl_3_column = [s3, s0, s1, s2]
 *
 * 接著分別計算：
 *   0e * column
 *   0b * sl_1_column
 *   0d * sl_2_column
 *   09 * sl_3_column
 *
 * 最後 XOR 起來即可得到 inverse MixColumns 結果。
 */
static void inv_mix_columns(uint8_t *state) {
    __m128i column;       /* 每個 column: [s0, s1, s2, s3] */
    __m128i sl_1_column;  /* 每個 column 左轉 1 byte: [s1, s2, s3, s0] */
    __m128i sl_2_column;  /* 每個 column 左轉 2 bytes: [s2, s3, s0, s1] */
    __m128i sl_3_column;  /* 每個 column 左轉 3 bytes: [s3, s0, s1, s2] */

    __m128i column_x2;
    __m128i column_x4;
    __m128i column_x8;

    __m128i sl_1_column_x2;
    __m128i sl_1_column_x4;
    __m128i sl_1_column_x8;

    __m128i sl_2_column_x2;
    __m128i sl_2_column_x4;
    __m128i sl_2_column_x8;

    __m128i sl_3_column_x2;
    __m128i sl_3_column_x4;
    __m128i sl_3_column_x8;

    __m128i mul_0e_column;
    __m128i mul_0b_sl_1_column;
    __m128i mul_0d_sl_2_column;
    __m128i mul_09_sl_3_column;
    __m128i result;

    /*
     * 讀入 16-byte AES state。
     * state 在記憶體中是 4 個連續的 4-byte column。
     */
    column = _mm_loadu_si128((const __m128i *)state);

    /*
     * 建立每個 column 的循環左轉版本。
     * 這樣同一個 byte 位置上就會分別對齊：
     *   s0, s1, s2, s3
     */
    sl_1_column = rotate_column_bytes_left(column);
    sl_2_column = rotate_column_bytes_left(sl_1_column);
    sl_3_column = rotate_column_bytes_left(sl_2_column);

    /*
     * 計算 0e * column。
     *
     * 在 AES GF(2^8) 中：
     *   0e*x = 14*x = 8*x ^ 4*x ^ 2*x
     */
    column_x2 = xtime_vec(column);
    column_x4 = xtime_vec(column_x2);
    column_x8 = xtime_vec(column_x4);
    mul_0e_column = _mm_xor_si128(column_x8, column_x4);
    mul_0e_column = _mm_xor_si128(mul_0e_column, column_x2);

    /*
     * 計算 0b * sl_1_column。
     *
     * 在 AES GF(2^8) 中：
     *   0b*x = 11*x = 8*x ^ 2*x ^ x
     */
    sl_1_column_x2 = xtime_vec(sl_1_column);
    sl_1_column_x4 = xtime_vec(sl_1_column_x2);
    sl_1_column_x8 = xtime_vec(sl_1_column_x4);
    mul_0b_sl_1_column = _mm_xor_si128(sl_1_column_x8, sl_1_column_x2);
    mul_0b_sl_1_column = _mm_xor_si128(mul_0b_sl_1_column, sl_1_column);

    /*
     * 計算 0d * sl_2_column。
     *
     * 在 AES GF(2^8) 中：
     *   0d*x = 13*x = 8*x ^ 4*x ^ x
     */
    sl_2_column_x2 = xtime_vec(sl_2_column);
    sl_2_column_x4 = xtime_vec(sl_2_column_x2);
    sl_2_column_x8 = xtime_vec(sl_2_column_x4);
    mul_0d_sl_2_column = _mm_xor_si128(sl_2_column_x8, sl_2_column_x4);
    mul_0d_sl_2_column = _mm_xor_si128(mul_0d_sl_2_column, sl_2_column);

    /*
     * 計算 09 * sl_3_column。
     *
     * 在 AES GF(2^8) 中：
     *   09*x = 9*x = 8*x ^ x
     */
    sl_3_column_x2 = xtime_vec(sl_3_column);
    sl_3_column_x4 = xtime_vec(sl_3_column_x2);
    sl_3_column_x8 = xtime_vec(sl_3_column_x4);
    mul_09_sl_3_column = _mm_xor_si128(sl_3_column_x8, sl_3_column);

    /*
     * 套用 InvMixColumns：
     *   result =
     *       0e * column
     *     ^ 0b * sl_1_column
     *     ^ 0d * sl_2_column
     *     ^ 09 * sl_3_column
     */
    result = _mm_xor_si128(mul_0e_column, mul_0b_sl_1_column);
    result = _mm_xor_si128(result, mul_0d_sl_2_column);
    result = _mm_xor_si128(result, mul_09_sl_3_column);

    /* 將 InvMixColumns 後的結果寫回 state。 */
    _mm_storeu_si128((__m128i *)state, result);
}

static void copy_previous_word(uint8_t word[4],
                               const uint8_t round_keys[AES_MAX_EXPANDED_KEY_SIZE],
                               size_t bytes_generated) {
    for (int i = 0; i < 4; i++) {
        word[i] = round_keys[bytes_generated - 4 + i];
    }
}

static void rotate_word_left(uint8_t word[4]) {
    uint8_t first = word[0];

    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = first;
}

static void sub_word(uint8_t word[4]) {
    for (int i = 0; i < 4; i++) {
        word[i] = sbox[word[i]];
    }
}

static void schedule_core(uint8_t word[4], int rcon_index) {
    rotate_word_left(word);
    sub_word(word);
    word[0] ^= rcon[rcon_index];
}

int aes_init(AESContext *ctx, const uint8_t *key, size_t key_size) {
    int bytes_generated;
    int expanded_key_size;
    int rcon_index = 1;
    int i;
    uint8_t temp[4];

    switch (key_size) {
        case AES_128_KEY_SIZE:
            ctx->key_words = 4;
            ctx->rounds = AES_128_ROUNDS;
            break;
        case AES_192_KEY_SIZE:
            ctx->key_words = 6;
            ctx->rounds = AES_192_ROUNDS;
            break;
        case AES_256_KEY_SIZE:
            ctx->key_words = 8;
            ctx->rounds = AES_256_ROUNDS;
            break;
        default:
            return 0;
    }

    ctx->key_size = key_size;
    expanded_key_size = (ctx->rounds + 1) * AES_BLOCK_SIZE;
    bytes_generated = (int)key_size;

    for (i = 0; i < (int)key_size; i++) {
        ctx->round_keys[i] = key[i];
    }

    size_t key_size_bytes = (size_t)ctx->key_words * 4;

    while (bytes_generated < expanded_key_size) {
        copy_previous_word(temp, ctx->round_keys, bytes_generated);

        if (bytes_generated % key_size_bytes == 0) {
            schedule_core(temp, rcon_index);
            rcon_index++;
        } else if (ctx->key_words > 6 &&
                    bytes_generated % key_size_bytes == AES_BLOCK_SIZE) {
            sub_word(temp);
        }

        for (i = 0; i < 4; i++) {
            ctx->round_keys[bytes_generated] =
                ctx->round_keys[bytes_generated - key_size_bytes] ^ temp[i];
            bytes_generated++;
        }
    }

    // while (bytes_generated < expanded_key_size) {
    //     for (i = 0; i < 4; i++) {
    //         temp[i] = ctx->round_keys[bytes_generated - 4 + i];
    //     }

    //     if (bytes_generated % (ctx->key_words * 4) == 0) {
    //         uint8_t first = temp[0];

    //         temp[0] = sbox[temp[1]] ^ rcon[rcon_index];
    //         temp[1] = sbox[temp[2]];
    //         temp[2] = sbox[temp[3]];
    //         temp[3] = sbox[first];

    //         rcon_index++;
    //     } else if (ctx->key_words > 6 && bytes_generated % (ctx->key_words * 4) == 16) {
    //         temp[0] = sbox[temp[0]];
    //         temp[1] = sbox[temp[1]];
    //         temp[2] = sbox[temp[2]];
    //         temp[3] = sbox[temp[3]];
    //     }

    //     for (i = 0; i < 4; i++) {
    //         ctx->round_keys[bytes_generated] =
    //             ctx->round_keys[bytes_generated - ctx->key_words * 4] ^ temp[i];
    //         bytes_generated++;
    //     }
    // }

    return 1;
}

/*
 * 加密單一 16-byte AES block。
 *
 * AES encryption round 流程：
 *   1. Initial AddRoundKey
 *   2. 前 rounds - 1 輪：
 *        SubBytes -> ShiftRows -> MixColumns -> AddRoundKey
 *   3. 最後一輪：
 *        SubBytes -> ShiftRows -> AddRoundKey
 *
 * 最後一輪不做 MixColumns，這是 AES 標準流程。
 */
static void aes_encrypt_block(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx) {

    uint8_t state[AES_BLOCK_SIZE];
    int round;

    /* 將輸入 block 複製到可修改的 state。 */
    memcpy(state, input, AES_BLOCK_SIZE);

    /* Round 0: 初始 AddRoundKey。 */
    add_round_key(state, ctx->round_keys);

    /* 中間 rounds：每輪包含 SubBytes、ShiftRows、MixColumns、AddRoundKey。 */
    for (round = 1; round < ctx->rounds; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, ctx->round_keys + round * AES_BLOCK_SIZE);
    }

    /* 最後一輪不執行 MixColumns。 */
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, ctx->round_keys + ctx->rounds * AES_BLOCK_SIZE);

    /* 將加密完成的 state 複製到 output。 */
    memcpy(output, state, AES_BLOCK_SIZE);
}



/*
 * 解密單一 16-byte AES block。
 *
 * AES decryption round 流程是 encryption 的反向：
 *   1. 先使用最後一組 round key 做 AddRoundKey
 *   2. 從 rounds - 1 倒數到 1：
 *        InvShiftRows -> InvSubBytes -> AddRoundKey -> InvMixColumns
 *   3. 最後：
 *        InvShiftRows -> InvSubBytes -> AddRoundKey
 *
 * 解密最後一步不做 InvMixColumns，對應加密最後一輪不做 MixColumns。
 */
static void aes_decrypt_block(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx
) {
    uint8_t state[AES_BLOCK_SIZE];
    int round;

    /* 將密文 block 複製到可修改的 state。 */
    memcpy(state, input, AES_BLOCK_SIZE);

    /* 解密一開始先套用最後一組 round key。 */
    add_round_key(state, ctx->round_keys + ctx->rounds * AES_BLOCK_SIZE);

    /* 從倒數第二組 round key 開始往回做 inverse rounds。 */
    for (round = ctx->rounds - 1; round > 0; round--) {
        inv_shift_rows(state);
        inv_sub_bytes(state);
        add_round_key(state, ctx->round_keys + round * AES_BLOCK_SIZE);
        inv_mix_columns(state);
    }

    /* 最後一輪不執行 InvMixColumns。 */
    inv_shift_rows(state);
    inv_sub_bytes(state);
    add_round_key(state, ctx->round_keys);

    /* 將解密完成的 state 複製到 output。 */
    memcpy(output, state, AES_BLOCK_SIZE);
}

/*
 * ECB mode encryption。
 *
 * ECB 會把輸入資料切成多個 16-byte block，
 * 每個 block 獨立呼叫 aes_encrypt_block() 加密。
 *
 * 注意：ECB 不使用 IV，也不會讓 block 之間互相影響。
 */
void aes_encrypt_ecb_normal(const uint8_t *input,
                            uint8_t *output,
                            size_t size,
                            const AESContext *ctx) {
    long block_count = (long)(size / AES_BLOCK_SIZE);

    #pragma omp parallel for
    for (long block = 0; block < block_count; block++) {
        size_t offset = (size_t)block * AES_BLOCK_SIZE;
        aes_encrypt_block(input + offset, output + offset, ctx);
    }
}

/*
 * ECB trace mode encryption。
 *
 * 這個版本支援 OpenMP 多執行緒：
 *   1. 每個 block 平行加密
 *   2. 每個 block 的 trace 先寫到自己的 TraceBuffer
 *   3. 平行區段結束後，再依照 block index 順序寫入同一個 trace 檔
 *
 * 這樣可以避免多個 thread 同時 fprintf 到同一個 FILE，
 * 造成 trace 順序混亂或檔案鎖競爭。
 */
void aes_encrypt_ecb_trace(const uint8_t *input,
                           uint8_t *output,
                           size_t size,
                           const AESContext *ctx,
                           FILE *trace) {
    long block_count = (long)(size / AES_BLOCK_SIZE);

    /*
     * 每個 block 的 trace 行數約為 ctx->rounds + 2：
     *   input
     *   add_round_key
     *   round_end x (rounds - 1)
     *   final_round_end
     *
     * 每行預留 160 bytes，避免 block index 變大時 buffer 不夠。
     */
    size_t trace_capacity = (size_t)(ctx->rounds + 2) * 160;

    TraceBuffer *traces;
    int failed = 0;

    /*
     * 為每個 block 準備一個 TraceBuffer。
     * traces[block] 只會被處理該 block 的 thread 使用，
     * 因此不需要 lock。
     */
    traces = (TraceBuffer *)calloc((size_t)block_count, sizeof(TraceBuffer));
    if (traces == NULL) {
        printf("錯誤：記憶體配置失敗\n");
        return;
    }

    /*
     * 平行處理每個 AES block。
     * schedule(static) 適合 ECB，因為每個 block 的工作量幾乎相同。
     */
    #pragma omp parallel for schedule(static)
    for (long block = 0; block < block_count; block++) {
        size_t offset = (size_t)block * AES_BLOCK_SIZE;

        /*
         * 每個 block 使用固定大小的 trace buffer。
         * 這比多次 realloc 或多 thread 直接寫檔更穩定。
         */
        traces[block].capacity = trace_capacity;
        traces[block].data = (char *)malloc(trace_capacity);
        traces[block].size = 0;

        /*
         * 加密該 block，並把每輪 state 寫入 traces[block]。
         * 如果 malloc 失敗或 trace buffer 寫入失敗，就標記 failed。
         */
        if (traces[block].data == NULL ||
            !aes_encrypt_block_with_trace(input + offset,
                                            output + offset,
                                            ctx,
                                            &traces[block],
                                            (size_t)block)) {
            /*
             * 多個 thread 可能同時設定 failed，
             * 用 atomic 避免 data race。
             */
            #pragma omp atomic write
            failed = 1;
        }
    }

    /*
     * 平行區段結束後，依照 block index 順序寫入 trace 檔。
     * 這樣 trace 檔會保持 block 0, block 1, block 2... 的順序。
     */
    if (!failed) {
        for (long block = 0; block < block_count; block++) {
            fwrite(traces[block].data, 1, traces[block].size, trace);
        }
    }

    /*
     * 釋放每個 block 的 trace buffer。
     */
    for (long block = 0; block < block_count; block++) {
        free(traces[block].data);
    }

    free(traces);
}

/*
 * 加密單一 16-byte AES block，並把中間 round state 寫入 TraceBuffer。
 *
 * 這個函式不直接寫 FILE，而是寫入傳入的 trace buffer。
 * 因此它可以安全地被多個 OpenMP thread 同時呼叫，
 * 只要每個 thread 使用不同的 TraceBuffer。
 */
static int aes_encrypt_block_with_trace(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx,
    TraceBuffer *trace,
    size_t block_index) {

    uint8_t state[AES_BLOCK_SIZE];
    int round;

    /*
     * 將 input block 複製到可修改的 state，
     * 並記錄最初的輸入狀態。
     */
    memcpy(state, input, AES_BLOCK_SIZE);
    if (!append_trace_state(trace, "input", block_index, 0, state)) {
        return 0;
    }

    /*
     * Round 0：Initial AddRoundKey。
     */
    add_round_key(state, ctx->round_keys);
    if (!append_trace_state(trace, "add_round_key", block_index, 0, state)) {
        return 0;
    }

    /*
     * 中間 rounds：
     *   SubBytes -> ShiftRows -> MixColumns -> AddRoundKey
     *
     * 每個 round 結束後記錄一次 state。
     */
    for (round = 1; round < ctx->rounds; round++) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, ctx->round_keys + round * AES_BLOCK_SIZE);

        if (!append_trace_state(trace, "round_end", block_index, round, state)) {
            return 0;
        }
    }

    /*
     * 最後一輪：
     *   SubBytes -> ShiftRows -> AddRoundKey
     *
     * AES 標準中最後一輪不執行 MixColumns。
     */
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, ctx->round_keys + ctx->rounds * AES_BLOCK_SIZE);

    if (!append_trace_state(trace, "final_round_end", block_index, ctx->rounds, state)) {
        return 0;
    }

    /*
     * 將加密結果寫入 output。
     */
    memcpy(output, state, AES_BLOCK_SIZE);
    return 1;
}

/*
 * ECB mode decryption。
 *
 * 將輸入密文切成多個 16-byte block，
 * 每個 block 獨立呼叫 aes_decrypt_block() 解密。
 */
void aes_decrypt_ecb_normal(const uint8_t *input,
                            uint8_t *output,
                            size_t size,
                            const AESContext *ctx) {
    long block_count = (long)(size / AES_BLOCK_SIZE);

    #pragma omp parallel for
    for (long block = 0; block < block_count; block++) {
        size_t offset = (size_t)block * AES_BLOCK_SIZE;
        aes_decrypt_block(input + offset, output + offset, ctx);
    }
}

/*
 * ECB mode decryption with trace。
 *
 * 與 aes_decrypt_ecb_normal() 相同，
 * 但會記錄每個 block 解密過程中的中間 state。
 */
/*
 * ECB trace mode decryption。
 *
 * 這個版本支援 OpenMP 多執行緒：
 *   1. 每個 block 平行解密
 *   2. 每個 block 的 trace 先寫到自己的 TraceBuffer
 *   3. 平行區段結束後，再依照 block index 順序寫入同一個 trace 檔
 *
 * 這樣可以避免多個 thread 同時 fprintf 到同一個 FILE，
 * 造成 trace 順序混亂或檔案鎖競爭。
 */
void aes_decrypt_ecb_trace(const uint8_t *input,
                           uint8_t *output,
                           size_t size,
                           const AESContext *ctx,
                           FILE *trace) {
    long block_count = (long)(size / AES_BLOCK_SIZE);

    /*
     * 每個 block 的 trace 行數約為 ctx->rounds + 2：
     *   input
     *   add_round_key
     *   round_end x (rounds - 1)
     *   final_round_end
     *
     * 每行預留 160 bytes，避免 block index 變大時 buffer 不夠。
     */
    size_t trace_capacity = (size_t)(ctx->rounds + 2) * 160;

    TraceBuffer *traces;
    int failed = 0;

    /*
     * 為每個 block 準備一個 TraceBuffer。
     * traces[block] 只會被處理該 block 的 thread 使用，
     * 因此不需要 lock。
     */
    traces = (TraceBuffer *)calloc((size_t)block_count, sizeof(TraceBuffer));
    if (traces == NULL) {
        printf("錯誤：記憶體配置失敗\n");
        return;
    }

    /*
     * 平行處理每個 AES block。
     * schedule(static) 適合 ECB，因為每個 block 的工作量幾乎相同。
     */
    #pragma omp parallel for schedule(static)
    for (long block = 0; block < block_count; block++) {
        size_t offset = (size_t)block * AES_BLOCK_SIZE;

        /*
         * 每個 block 使用固定大小的 trace buffer。
         * 這比多次 realloc 或多 thread 直接寫檔更穩定。
         */
        traces[block].capacity = trace_capacity;
        traces[block].data = (char *)malloc(trace_capacity);
        traces[block].size = 0;

        /*
         * 解密該 block，並把每輪 state 寫入 traces[block]。
         * 如果 malloc 失敗或 trace buffer 寫入失敗，就標記 failed。
         */
        if (traces[block].data == NULL ||
            !aes_decrypt_block_with_trace(input + offset,
                                            output + offset,
                                            ctx,
                                            &traces[block],
                                            (size_t)block)) {
            /*
             * 多個 thread 可能同時設定 failed，
             * 用 atomic 避免 data race。
             */
            #pragma omp atomic write
            failed = 1;
        }
    }

    /*
     * 平行區段結束後，依照 block index 順序寫入 trace 檔。
     * 這樣 trace 檔會保持 block 0, block 1, block 2... 的順序。
     */
    if (!failed) {
        for (long block = 0; block < block_count; block++) {
            fwrite(traces[block].data, 1, traces[block].size, trace);
        }
    }

    /*
     * 釋放每個 block 的 trace buffer。
     */
    for (long block = 0; block < block_count; block++) {
        free(traces[block].data);
    }

    free(traces);
}

/*
 * 解密單一 16-byte AES block，並把中間 round state 寫入 TraceBuffer。
 *
 * 這個函式不直接寫 FILE，而是寫入傳入的 trace buffer。
 * 因此它可以安全地被多個 OpenMP thread 同時呼叫，
 * 只要每個 thread 使用不同的 TraceBuffer。
 */
static int aes_decrypt_block_with_trace(
    const uint8_t input[AES_BLOCK_SIZE],
    uint8_t output[AES_BLOCK_SIZE],
    const AESContext *ctx,
    TraceBuffer *trace,
    size_t block_index) {

    uint8_t state[AES_BLOCK_SIZE];
    int round;

    /*
     * 將 input ciphertext block 複製到可修改的 state，
     * 並記錄最初的密文狀態。
     *
     * 解密 trace 從 ctx->rounds 開始標記，
     * 因為解密流程會從最後一輪 round key 往前推回 round 0。
     */
    memcpy(state, input, AES_BLOCK_SIZE);
    if (!append_trace_state(trace, "input", block_index, ctx->rounds, state)) {
        return 0;
    }

    /*
     * 解密第一步：先套用最後一組 round key。
     */
    add_round_key(state, ctx->round_keys + ctx->rounds * AES_BLOCK_SIZE);
    if (!append_trace_state(trace, "add_round_key", block_index, ctx->rounds, state)) {
        return 0;
    }

    /*
     * 中間 inverse rounds：
     *   InvShiftRows -> InvSubBytes -> AddRoundKey -> InvMixColumns
     *
     * round 從 ctx->rounds - 1 一路倒數到 1。
     * 每個 inverse round 結束後記錄一次 state。
     */
    for (round = ctx->rounds - 1; round > 0; round--) {
        inv_shift_rows(state);
        inv_sub_bytes(state);
        add_round_key(state, ctx->round_keys + round * AES_BLOCK_SIZE);
        inv_mix_columns(state);

        if (!append_trace_state(trace, "round_end", block_index, round, state)) {
            return 0;
        }
    }

    /*
     * 最後一輪：
     *   InvShiftRows -> InvSubBytes -> AddRoundKey
     *
     * 對應加密最後一輪不做 MixColumns，
     * 解密最後一輪也不做 InvMixColumns。
     */
    inv_shift_rows(state);
    inv_sub_bytes(state);
    add_round_key(state, ctx->round_keys);

    if (!append_trace_state(trace, "final_round_end", block_index, 0, state)) {
        return 0;
    }

    /*
     * 將解密後的 plaintext block 寫入 output。
     */
    memcpy(output, state, AES_BLOCK_SIZE);
    return 1;
}


/*
 * 將一筆 AES state trace 寫入指定的 TraceBuffer。
 *
 * 這個函式是 trace_state() 的 buffer 版本：
 *   - 不直接寫入 FILE
 *   - 只把文字寫到目前 block 專屬的 TraceBuffer
 *
 * 這樣每個 OpenMP thread 都可以安全地寫自己的 buffer，
 * 不會多個 thread 同時寫同一個 trace 檔。
 *
 * 輸出格式：
 *   block=<block_index> round=<round> <label> <state_hex>
 *
 * 範例：
 *   block=0 round=01 round_end        00112233445566778899aabbccddeeff
 */
static int append_trace_state(TraceBuffer *trace,
                              const char *label,
                              size_t block_index,
                              int round,
                              const uint8_t state[AES_BLOCK_SIZE]) {
    int written;
    size_t remaining;
    char *cursor;

    /*
     * 找到目前 buffer 還能寫入的位置。
     * trace->size 表示目前已經使用的 bytes。
     * trace->capacity 表示整個 buffer 的總容量。
     */
    remaining = trace->capacity - trace->size;
    cursor = trace->data + trace->size;

    /*
     * 先寫入 trace line 的前半段：
     *   block=<index> round=<round> <label>
     */
    written = snprintf(cursor, remaining,
                       "block=%zu round=%02d %-16s ",
                       block_index,
                       round,
                       label);
    if (written < 0 || (size_t)written >= remaining) {
        return 0;
    }

    trace->size += (size_t)written;

    /*
     * 接著把 16-byte AES state 轉成 32 個 hex 字元。
     * 每個 byte 會輸出成兩位十六進位，例如 0a、ff。
     */
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        remaining = trace->capacity - trace->size;
        cursor = trace->data + trace->size;

        written = snprintf(cursor, remaining, "%02x", state[i]);
        if (written < 0 || (size_t)written >= remaining) {
            return 0;
        }

        trace->size += (size_t)written;
    }

    /*
     * 一筆 state trace 結束，補上換行。
     */
    remaining = trace->capacity - trace->size;
    cursor = trace->data + trace->size;

    written = snprintf(cursor, remaining, "\n");
    if (written < 0 || (size_t)written >= remaining) {
        return 0;
    }

    trace->size += (size_t)written;

    return 1;
}
