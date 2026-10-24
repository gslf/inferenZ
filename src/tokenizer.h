// Byte-level BPE tokenizer matching the MiniMax-M3.1 tokenizer.json:
// NFC normalizer -> o200k-style regex split -> byte-level BPE merges, with added tokens.
#ifndef INZ_TOKENIZER_H
#define INZ_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct tokenizer tokenizer;

enum { TOK_SPLIT_O200K = 0, TOK_SPLIT_QWEN = 1 };
void tok_set_split(tokenizer *t, int split);
int tok_get_split(const tokenizer *t);

// Build from a HF tokenizer.json (converter path).
tokenizer *tok_load_hf_json(const char *path, char *err, size_t errlen);
// Build from arrays stored in the GGUF (runtime path). pieces are raw bytes (not byte-level encoded).
tokenizer *tok_create(int n_vocab, const char *const *pieces, const uint32_t *piece_len, int n_merges,
                      const int32_t *merge_a, const int32_t *merge_b, const int32_t *merge_res,
                      int n_added, const int32_t *added_ids, const char *const *added_str, const uint8_t *added_special);
void tok_free(tokenizer *t);

int tok_vocab(const tokenizer *t);
// Encode UTF-8 text. parse_special: recognize added-token strings in the text.
// Returns number of tokens (may exceed cap; then only cap were written).
int tok_encode(tokenizer *t, const char *text, size_t len, bool parse_special, int32_t *out, int cap);
// Raw bytes of one token.
const char *tok_piece(const tokenizer *t, int id, size_t *len);
int tok_find(const tokenizer *t, const char *str);  // exact piece / added token lookup, -1 if none
bool tok_is_special(const tokenizer *t, int id);

// Export for GGUF writing (raw bytes per piece, merges as id triples, added tokens).
typedef struct {
    int n_vocab;
    const char **pieces;
    uint32_t *piece_len;
    int n_merges;
    int32_t *merge_a, *merge_b, *merge_res;
    int n_added;
    int32_t *added_ids;
    const char **added_str;
    uint8_t *added_special;
} tok_export;
void tok_get_export(const tokenizer *t, tok_export *e);

// Incremental UTF-8 safe detokenizer: accumulates bytes and returns only complete characters.
typedef struct {
    char pend[8];
    int npend;
} tok_stream;
// Appends printable text for `id` into out (returns bytes written, out must hold >= 64 + piece len).
size_t tok_stream_push(const tokenizer *t, tok_stream *s, int id, char *out, size_t cap);

// Unicode helpers (exported for tests)
size_t inz_nfc(const char *in, size_t len, char **out);  // returns new length, *out malloc'ed

#endif
