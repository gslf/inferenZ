// Read-only access to a sharded safetensors checkpoint (possibly still downloading).
#ifndef INZ_SAFETENSORS_H
#define INZ_SAFETENSORS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum { ST_F32, ST_F16, ST_BF16, ST_F8_E4M3, ST_U8, ST_I8, ST_I32, ST_I64, ST_F64, ST_UNKNOWN } st_dtype;

typedef struct {
    char *name;
    st_dtype dtype;
    int ndim;
    int64_t shape[4];
    int file;          // index into st_model.files
    uint64_t start;    // absolute byte offset in file
    uint64_t nbytes;
} st_tensor;

typedef struct {
    int nfiles;
    char **paths;
    int *fds;
    // availability journal (hfget): per-file sorted list of complete ranges
    uint64_t **ranges;  // [file] -> pairs (start,end) sorted & merged
    int *nranges;
    bool have_journal;
    int ntensors;
    st_tensor *tensors;
    // name lookup
    int32_t *htab;
    uint32_t hmask;
} st_model;

// dir must contain model.safetensors.index.json (or a single model.safetensors).
st_model *st_open(const char *dir, char *err, size_t errlen);
void st_close(st_model *m);
const st_tensor *st_find(const st_model *m, const char *name);
// true if the tensor bytes are fully present on disk (journal aware)
bool st_available(const st_model *m, const st_tensor *t);
// re-read the download journal (cheap)
void st_refresh(st_model *m);
// read raw bytes
bool st_read(const st_model *m, const st_tensor *t, void *dst);
int64_t st_numel(const st_tensor *t);
const char *st_dtype_name(st_dtype d);

#endif
