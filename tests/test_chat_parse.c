// Streaming output parser: feed model outputs one byte at a time and check reasoning / content / tool calls.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/chat.h"

static int fails;

static void run(const char *name, const char *out, bool thinking, const char *tools_json, const char *want_r,
                const char *want_c, const char *want_tools_substr) {
    char err[64];
    jval *tools = tools_json ? json_parse(tools_json, strlen(tools_json), err, sizeof err) : NULL;
    chat_parser p;
    chat_parser_init(&p, thinking, tools);
    char r[4096] = "", c[4096] = "";
    const char *dr, *dc;
    size_t nr, nc;
    for (size_t i = 0; out[i]; i++) {
        chat_parser_feed(&p, out + i, 1, &dr, &nr, &dc, &nc);
        strncat(r, dr, nr);
        strncat(c, dc, nc);
    }
    char *tc = chat_parser_finish(&p, &dr, &nr, &dc, &nc);
    strncat(r, dr, nr);
    strncat(c, dc, nc);
    bool ok = !strcmp(r, want_r) && !strcmp(c, want_c) &&
              (want_tools_substr ? tc && strstr(tc, want_tools_substr) : tc == NULL);
    printf("%-28s %s\n", name, ok ? "ok" : "FAIL");
    if (!ok) { printf("  reasoning [%s]\n  content   [%s]\n  tools     %s\n", r, c, tc ? tc : "(none)"); fails++; }
    free(tc);
    chat_parser_free(&p);
    json_free(tools);
}

int main(void) {
    const char *tools = "[{\"type\":\"function\",\"function\":{\"name\":\"run_c\",\"parameters\":{\"type\":\"object\",\"properties\":"
                        "{\"source\":{\"type\":\"string\"},\"timeout\":{\"type\":\"number\"},\"args\":{\"type\":\"array\"}}}}}]";
    run("think + answer", "Let me think.\nOK\n</think>\n\nThe answer is 42.", true, NULL, "Let me think.\nOK", "The answer is 42.", NULL);
    run("no thinking", "Direct answer.\n\nSecond paragraph.", false, NULL, "", "Direct answer.\n\nSecond paragraph.", NULL);
    run("tool call with C source",
        "plan</think>I'll compile it.<tool_call>run_c<arg_key>source</arg_key><arg_value>#include <stdio.h>\n"
        "int main(void) { puts(\"hi\\n\"); return 0; }\n</arg_value><arg_key>timeout</arg_key><arg_value>2.5</arg_value>"
        "<arg_key>args</arg_key><arg_value>[\"-O2\", \"x\"]</arg_value></tool_call>",
        true, tools, "plan", "I'll compile it.",
        "\"arguments\": \"{\\\"source\\\": \\\"#include <stdio.h>\\\\nint main(void) { puts(\\\\\\\"hi\\\\\\\\n\\\\\\\"); return 0; }\\\", "
        "\\\"timeout\\\": 2.5, \\\"args\\\": [\\\"-O2\\\", \\\"x\\\"]}\"");
    run("string param that looks numeric",
        "x</think><tool_call>run_c<arg_key>source</arg_key><arg_value>123</arg_value></tool_call>",
        true, tools, "x", "", "\\\"source\\\": \\\"123\\\"");
    run("two tool calls",
        "</think><tool_call>run_c<arg_key>source</arg_key><arg_value>a</arg_value></tool_call>"
        "<tool_call>run_c<arg_key>source</arg_key><arg_value>b</arg_value></tool_call>",
        true, tools, "", "", "\"index\": 1");
    run("unknown tool, json value",
        "</think><tool_call>other<arg_key>n</arg_key><arg_value>7</arg_value><arg_key>s</arg_key><arg_value>hello</arg_value></tool_call>",
        true, tools, "", "", "{\\\"n\\\": 7, \\\"s\\\": \\\"hello\\\"}");
    run("tool call without arguments", "</think><tool_call>ping</tool_call>", true, tools, "", "", "\"name\": \"ping\", \"arguments\": \"{}\"");
    run("newlines inside reasoning", "a\n\nb\n</think>\nc", true, NULL, "a\n\nb", "c", NULL);
    printf("%s\n", fails ? "FAILURES" : "all parser tests passed");
    return fails != 0;
}
