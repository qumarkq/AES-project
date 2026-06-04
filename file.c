#include "file.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int read_file(const char *filename, FileBuffer *buffer) {
    int fd;
    struct stat file_stat;
    size_t bytes_read = 0;

    buffer->data = NULL;
    buffer->size = 0;

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        printf("錯誤：無法開啟檔案，請確認權限或檔案是否存在 %s\n", filename);
        return 0;
    }

    if (fstat(fd, &file_stat) != 0) {
        printf("錯誤：無法取得檔案大小 %s\n", filename);
        close(fd);
        return 0;
    }

    if (file_stat.st_size < 0) {
        printf("錯誤：檔案大小無效 %s\n", filename);
        close(fd);
        return 0;
    }

    buffer->size = (size_t)file_stat.st_size;
    if (buffer->size == 0) {
        close(fd);
        return 1;
    }

    buffer->data = (uint8_t *)malloc(buffer->size);
    if (buffer->data == NULL) {
        printf("錯誤：記憶體配置失敗\n");
        close(fd);
        buffer->size = 0;
        return 0;
    }

    while (bytes_read < buffer->size) {
        ssize_t result = read(fd, buffer->data + bytes_read,
                              buffer->size - bytes_read);
        if (result <= 0) {
            printf("錯誤：無法讀取檔案 %s\n", filename);
            free(buffer->data);
            buffer->data = NULL;
            buffer->size = 0;
            close(fd);
            return 0;
        }

        bytes_read += (size_t)result;
    }

    close(fd);
    return 1;
}

int write_file(const char *filename, const uint8_t *data, size_t size) {
    int fd;
    size_t bytes_written = 0;

    fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        printf("錯誤：無法開啟輸出檔案 %s\n", filename);
        return 0;
    }

    while (bytes_written < size) {
        ssize_t result = write(fd, data + bytes_written,
                               size - bytes_written);
        if (result <= 0) {
            printf("錯誤：無法寫入檔案 %s\n", filename);
            close(fd);
            return 0;
        }

        bytes_written += (size_t)result;
    }

    close(fd);
    return 1;
}

void free_file_buffer(FileBuffer *buffer) {
    free(buffer->data);
    buffer->data = NULL;
    buffer->size = 0;
}
