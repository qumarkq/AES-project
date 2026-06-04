#ifndef FILE_H
#define FILE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t size;
} FileBuffer;

int read_file(const char *filename, FileBuffer *buffer);
int write_file(const char *filename, const uint8_t *data, size_t size);
void free_file_buffer(FileBuffer *buffer);

#endif
