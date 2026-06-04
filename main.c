#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "aes.h"
#include "file.h"

typedef enum {
    MODE_ENCRYPT,
    MODE_DECRYPT
} Mode;

typedef struct {
    Mode mode;
    const char *input_file;
    const char *key_file;
    const char *output_file;
    const char *trace_file;
    int trace_enabled;
} ProgramOptions;

void print_usage(const char *program_name);

int parse_arguments(int argc,
                    char *argv[],
                    ProgramOptions *options);

void print_section_header(const char *title);

void print_options(const ProgramOptions *options);

int validate_input_and_key(const FileBuffer *input,
                           const FileBuffer *key);

FILE *open_trace_file(const ProgramOptions *options);

double run_aes(const ProgramOptions *options,
               const AESContext *aes_ctx,
               const FileBuffer *input,
               uint8_t *output,
               FILE *trace);

void print_performance(size_t processed_bytes,
                       double elapsed_seconds);


/*
 * 程式進入點。
 *
 * 執行流程：
 *   1. 解析命令列參數。
 *   2. 讀取輸入檔案與 key 檔案。
 *   3. 檢查輸入大小與 key 長度是否合法。
 *   4. 初始化 AES context。
 *   5. 配置輸出 buffer。
 *   6. 視需要開啟 trace 檔案。
 *   7. 執行 AES 加密或解密並印出效能。
 *   8. 寫出結果並釋放所有資源。
 */
int main(int argc, char *argv[]) {
    ProgramOptions options;
    FileBuffer input;
    FileBuffer key;
    AESContext aes_ctx;
    uint8_t *output = NULL;
    FILE *trace = NULL;
    double elapsed_seconds;

    /* 解析 -e / -d、可選的 -x trace_file，以及三個檔案路徑。 */
    if (!parse_arguments(argc, argv, &options)) {
        print_usage(argv[0]);
        return 1;
    }

    /* 印出本次執行設定，方便確認 mode 與檔案路徑。 */
    print_options(&options);

    /* 讀取待加密或待解密的輸入檔案。 */
    if (!read_file(options.input_file, &input)) {
        return 1;
    }

    /* 讀取 AES key 檔案；若失敗，需要釋放已讀入的 input。 */
    if (!read_file(options.key_file, &key)) {
        free_file_buffer(&input);
        return 1;
    }

    /* 檢查 input 是否為 block size 倍數，以及 key 是否為合法 AES 長度。 */
    if (!validate_input_and_key(&input, &key)) {
        free_file_buffer(&input);
        free_file_buffer(&key);
        return 1;
    }

    /* 根據 key 長度初始化 AES-128 / AES-192 / AES-256 context。 */
    if (!aes_init(&aes_ctx, key.data, key.size)) {
        printf("錯誤：AES key 長度無效\n");
        free_file_buffer(&input);
        free_file_buffer(&key);
        return 1;
    }

    /* 輸出大小與輸入大小相同，因為本程式只處理完整 AES block，不做 padding。 */
    output = (uint8_t *)malloc(input.size);
    if (output == NULL) {
        printf("錯誤：記憶體配置失敗\n");
        free_file_buffer(&input);
        free_file_buffer(&key);
        return 1;
    }

    /* 若命令列有指定 -x，則開啟 trace 檔案用來記錄每個 round 的 state。 */
    trace = open_trace_file(&options);
    if (options.trace_enabled && trace == NULL) {
        free(output);
        free_file_buffer(&input);
        free_file_buffer(&key);
        return 1;
    }

    /* 執行 AES 加密或解密，並量測 AES 核心處理時間。 */
    elapsed_seconds = run_aes(&options, &aes_ctx, &input, output, trace);
    print_performance(input.size, elapsed_seconds);

    /* trace 已寫完，先關閉檔案，避免後續流程遺漏 flush。 */
    if (trace != NULL) {
        fclose(trace);
        trace = NULL;
    }

    /* 將加密或解密結果寫到輸出檔案。 */
    if (!write_file(options.output_file, output, input.size)) {
        free(output);
        free_file_buffer(&input);
        free_file_buffer(&key);
        return 1;
    }

    /* 釋放所有動態配置的記憶體。 */
    free(output);
    free_file_buffer(&input);
    free_file_buffer(&key);

    return 0;
}


/*
 * 印出程式使用方式。
 */
void print_usage(const char *program_name) {
    printf("Usage:\n");
    printf("  %s -e [-x trace_file] input_file key_file output_file\n", program_name);
    printf("  %s -d [-x trace_file] input_file key_file output_file\n", program_name);
}

/*
 * 解析命令列參數。
 *
 * 支援格式：
 *   -e [-x trace_file] input_file key_file output_file
 *   -d [-x trace_file] input_file key_file output_file
 *
 * 回傳 1 表示解析成功，0 表示參數格式錯誤。
 */
int parse_arguments(int argc, char *argv[], ProgramOptions *options) {
    int index = 2;

    if (argc != 5 && argc != 7) {
        return 0;
    }

    options->trace_enabled = 0;
    options->trace_file = NULL;

    if (strcmp(argv[1], "-e") == 0) {
        options->mode = MODE_ENCRYPT;
    } else if (strcmp(argv[1], "-d") == 0) {
        options->mode = MODE_DECRYPT;
    } else {
        return 0;
    }

    if (argc == 7) {
        if (strcmp(argv[index], "-x") != 0) {
            return 0;
        }

        options->trace_enabled = 1;
        options->trace_file = argv[index + 1];
        index += 2;
    }

    options->input_file = argv[index];
    options->key_file = argv[index + 1];
    options->output_file = argv[index + 2];

    return 1;
}

/*
 * 印出置中的區段標題。
 *
 * width 表示整行目標寬度，標題左右會用 '=' 補齊。
 */
void print_section_header(const char *title) {
    const int width = 48;
    int title_length = (int)strlen(title);
    int left_padding = 0;
    int right_padding = 0;

    if (title_length + 2 < width) {
        left_padding = (width - title_length - 2) / 2;
        right_padding = width - title_length - 2 - left_padding;
    }

    printf("%.*s %s %.*s\n",
           left_padding,
           "================================================",
           title,
           right_padding,
           "================================================");
}

/*
 * 印出本次執行的模式、輸入檔、key 檔、輸出檔與 trace 檔設定。
 */
void print_options(const ProgramOptions *options) {
    print_section_header("Run configuration");
    printf("Mode        : %s\n", options->mode == MODE_ENCRYPT ? "encrypt" : "decrypt");
    printf("Input file  : %s\n", options->input_file);
    printf("Key file    : %s\n", options->key_file);
    printf("Output file : %s\n", options->output_file);

    if (options->trace_enabled) {
        printf("Trace file  : %s\n", options->trace_file);
    }

    printf("\n");
}

/*
 * 印出輸入檔案與 key 的大小，並檢查格式是否合法。
 *
 * key 長度只接受：
 *   16 bytes = 128 bits
 *   24 bytes = 192 bits
 *   32 bytes = 256 bits
 *
 * 輸入檔案大小必須非零，且必須是 AES_BLOCK_SIZE 的倍數。
 *
 * 回傳 1 表示合法，0 表示不合法。
 */
int validate_input_and_key(const FileBuffer *input, const FileBuffer *key) {
    print_section_header("Input summary");
    printf("Input size  : %zu bytes\n", input->size);
    printf("Key size    : %zu bytes (%zu bits)\n", key->size, key->size * 8);
    printf("\n");

    if (key->size != AES_128_KEY_SIZE &&
        key->size != AES_192_KEY_SIZE &&
        key->size != AES_256_KEY_SIZE) {
        printf("錯誤：key 長度必須是 128、192 或 256 bits（16、24 或 32 bytes）\n");
        return 0;
    }

    if (input->size == 0 || input->size % AES_BLOCK_SIZE != 0) {
        printf("錯誤：輸入檔案大小必須是非零，且為 %d bytes 的倍數\n", AES_BLOCK_SIZE);
        return 0;
    }

    return 1;
}

/*
 * 根據 options 判斷是否需要開啟 trace 檔案。
 *
 * 沒有啟用 trace 時直接回傳 NULL。
 * 有啟用 trace 但開檔失敗時，也回傳 NULL 並印出錯誤訊息。
 */
FILE *open_trace_file(const ProgramOptions *options) {
    FILE *trace;

    if (!options->trace_enabled) {
        return NULL;
    }

    trace = fopen(options->trace_file, "w");
    if (trace == NULL) {
        printf("錯誤：無法開啟 trace 檔案 %s\n", options->trace_file);
    }

    return trace;
}

/*
 * 執行 AES ECB 加密或解密，並測量 AES 處理時間。
 *
 * trace == NULL 時使用 normal 版本。
 * trace != NULL 時使用 trace 版本，會額外輸出每個 round 的 state。
 *
 * 回傳值是 AES 執行花費的秒數。
 */
double run_aes(const ProgramOptions *options,
               const AESContext *aes_ctx,
               const FileBuffer *input,
               uint8_t *output,
               FILE *trace) {
    double start_time;
    double end_time;

    start_time = omp_get_wtime();

    switch (options->mode) {
        case MODE_ENCRYPT:
            if (trace == NULL) {
                aes_encrypt_ecb_normal(input->data, output, input->size, aes_ctx);
            } else {
                aes_encrypt_ecb_trace(input->data, output, input->size, aes_ctx, trace);
            }
            break;

        case MODE_DECRYPT:
            if (trace == NULL) {
                aes_decrypt_ecb_normal(input->data, output, input->size, aes_ctx);
            } else {
                aes_decrypt_ecb_trace(input->data, output, input->size, aes_ctx, trace);
            }
            break;
    }

    end_time = omp_get_wtime();
    return end_time - start_time;
}

/*
 * 印出 AES 處理資料量、耗時與效能。
 *
 * Performance 使用 bytes/second。
 */
void print_performance(size_t processed_bytes, double elapsed_seconds) {
    double bytes_per_second = 0.0;

    if (elapsed_seconds > 0.0) {
        bytes_per_second = (double)processed_bytes / elapsed_seconds;
    }

    print_section_header("Performance");
    printf("Processed   : %zu bytes\n", processed_bytes);
    printf("Elapsed     : %.9f seconds\n", elapsed_seconds);
    printf("Performance : %.2f bytes/second\n", bytes_per_second);
}
