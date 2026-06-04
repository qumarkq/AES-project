# 編譯器
CC = gcc

# 輸出的執行檔名稱
TARGET = aes

# 原始碼檔案
SRC = main.c aes.c file.c

# make run 的預設執行參數
# 可以用下面方式覆蓋：
#   make run THREADS=1
#   make run THREADS=4 INPUT=pt_vec.bin KEY=key192_vec.bin OUTPUT=out192.dat
#   make run TRACE=trace.txt
#   make run MODE=-d TRACE=trace_decrypt.txt INPUT=測試資料/ciphertext1.dat OUTPUT=out_de.dat
THREADS ?= 8
MODE ?= -e
INPUT ?= 測試資料/ciphertext2.dat
KEY ?= 測試資料/key128b.bin
OUTPUT ?= out_e.dat

# Trace 輸出檔案名稱
# 預設為空，表示不啟用 trace mode。
# 如果指定 TRACE，例如：
#   make run TRACE=trace.txt
# Makefile 會自動在執行參數中加入：
#   -x trace.txt
TRACE ?= trace_parallel.txt
TRACE_ARGS = $(if $(TRACE),-x $(TRACE),)

# 共用的 C 標準與警告選項
# -std=c11   : 使用 C11 標準
# -Wall      : 開啟常見警告
# -Wextra    : 開啟額外警告
# -pedantic  : 對非標準 C 寫法提出警告
WARN_FLAGS = -std=c11 -Wall -Wextra -pedantic

# 一般效能測試用的最佳化選項
# -O3        : 開啟高等級最佳化
OPT_FLAGS = -O3

# SIMD 與 OpenMP 加速選項
# -DUSE_SIMD : 啟用 aes.c 裡的 SIMD-only 程式路徑
# -mssse3    : 啟用 SSSE3 intrinsic，例如 _mm_shuffle_epi8
# -fopenmp   : 啟用 OpenMP parallel loop，並連結 OpenMP runtime
ACCEL_FLAGS = -DUSE_SIMD -mssse3 -fopenmp

# 預設編譯選項：警告 + 最佳化 + SIMD/OpenMP
CFLAGS = $(WARN_FLAGS) $(OPT_FLAGS) $(ACCEL_FLAGS)

.PHONY: all run clean

# 預設目標：建立 SIMD + OpenMP 的 AES 執行檔
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# 執行程式，並用 THREADS 指定 OpenMP 執行緒數量。
# 如果 TRACE 不為空，會自動加上 -x <trace file> 來輸出每輪 AES state。
# 使用範例：
#   make run
#   make run THREADS=1
#   make run THREADS=8 INPUT=big_plain.dat KEY=測試資料/key128b.bin OUTPUT=big_out.dat
#   make run TRACE=trace.txt INPUT=測試資料/plaintext1.dat OUTPUT=out_trace.dat
run: $(TARGET)
	@OMP_NUM_THREADS=$(THREADS) ./$(TARGET) $(MODE) $(TRACE_ARGS) $(INPUT) $(KEY) $(OUTPUT) || true

# 清除產生的執行檔與 profiling 輸出
clean:
	rm -f $(TARGET)
