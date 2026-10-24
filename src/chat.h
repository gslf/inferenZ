// Qwen3.8-Flash-Next chat template (C port of chat_template.jinja) and output parsing
// (reasoning / content / tool calls) for the OpenAI-compatible API.
#ifndef INZ_CHAT_H
#define INZ_CHAT_H

#include <stdbool.h>
#include <stddef.h>

#include "json.h"

// messages: JSON array of {role, content, reasoning_content?, tool_calls?}; tools: JSON array or NULL.
// effort: NULL (template default: xhigh), "xhigh" | "high" | "max" -> xhigh, "medium", "low",
// "none" | "minimal" | "off" -> thinking disabled. Returns malloc'ed prompt text.
char *chat_render(const jval *messages, const jval *tools, const char *effort, bool add_generation_prompt);
// true when the rendered generation prompt leaves the model inside a <think> block
bool chat_thinking_enabled(const char *effort);

// Incremental parser of the model's raw output.
typedef enum { CP_START, CP_THINK, CP_CONTENT, CP_TOOL } chat_phase;

typedef struct {
    chat_phase phase;
    char *raw;            // full raw text
    size_t raw_len, raw_cap;
    size_t scanned;       // bytes of raw already classified
    // accumulated outputs
    char *reasoning; size_t reasoning_len, reasoning_cap;
    char *content; size_t content_len, content_cap;
    char *tool_raw; size_t tool_len, tool_cap;
    bool skip_nl;         // drop newlines right after </think>
    const jval *tools;    // request tools (parameter types for argument conversion), may be NULL
} chat_parser;

// thinking: the prompt ended inside <think> (see chat_thinking_enabled)
void chat_parser_init(chat_parser *p, bool thinking, const jval *tools);
void chat_parser_free(chat_parser *p);
// Feed newly generated text. Returns deltas (pointers into internal buffers, valid until next call).
void chat_parser_feed(chat_parser *p, const char *text, size_t n,
                      const char **d_reasoning, size_t *n_reasoning, const char **d_content, size_t *n_content);
// Finish: flush held-back text. Tool calls (if any) are returned as an OpenAI tool_calls JSON array string
// (malloc'ed) or NULL.
char *chat_parser_finish(chat_parser *p, const char **d_reasoning, size_t *n_reasoning, const char **d_content,
                         size_t *n_content);

#endif
