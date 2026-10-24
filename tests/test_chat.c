// Render chat templates for JSON test cases read from stdin (one JSON object per line):
// {"messages": [...], "tools": [...], "effort": "...", "add_generation_prompt": true}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/chat.h"
int main(void) {
    char *line = NULL; size_t cap = 0; ssize_t n;
    while ((n = getline(&line, &cap, stdin)) > 0) {
        char err[128];
        jval *v = json_parse(line, (size_t)n, err, sizeof err);
        if (!v) { printf("PARSE ERROR %s\n", err); continue; }
        char *s = chat_render(json_get(v, "messages"), json_get(v, "tools"), json_str(json_get(v, "effort"), NULL),
                              json_bool(json_get(v, "add_generation_prompt"), true));
        fwrite(s, 1, strlen(s), stdout);
        fputc(0, stdout);
        free(s);
        json_free(v);
    }
    return 0;
}
