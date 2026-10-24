# Compare the C chat template (tests/test_chat) with the official Jinja template of Qwen3.8-Flash-Next,
# rendered the way transformers does (sandboxed Jinja, trim_blocks, lstrip_blocks, Python-style tojson).
import json, subprocess, sys
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
tpl_path, exe = sys.argv[1], sys.argv[2]
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def raise_exception(m): raise RuntimeError(m)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(tpl_path).read())
run_c = {"type": "function", "function": {"name": "run_c", "description": "Compile and run a C program è",
         "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "args": {"type": "array", "items": {"type": "string"}},
                        "timeout": {"type": "number"}}, "required": ["source"]}}}
run_js = {"type": "function", "function": {"name": "run_js", "description": "Run JavaScript with node", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}
code = '#include <stdio.h>\nint main(void) {\n    printf("%d\\n", 6 * 7);\n    return 0;\n}\n'
cases = [
  {"messages": [{"role": "user", "content": "Ciao!"}]},
  {"messages": [{"role": "system", "content": "  Be terse.  "}, {"role": "user", "content": "2+2?"}], "effort": "low"},
  {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}], "effort": "medium"},
  {"messages": [{"role": "user", "content": "x"}], "effort": "medium"},
  {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "x"}]},
  {"messages": [{"role": "user", "content": "no thinking"}], "effort": "none"},
  {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "answer one", "reasoning_content": "\nhmm\n"},
                {"role": "user", "content": "q2"}]},
  {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "plain"},
                {"role": "user", "content": [{"type": "text", "text": "multi"}, {"type": "text", "text": " part"}]}]},
  {"messages": [{"role": "system", "content": "You are a C expert."}, {"role": "user", "content": "Write and test hello"},
                {"role": "assistant", "content": "I will run it.", "reasoning_content": "plan", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "run_c", "arguments": {"source": code, "args": ["-v", "x y"], "timeout": 2.5}}}]},
                {"role": "tool", "content": "42\n"}, {"role": "user", "content": "thanks"}], "tools": [run_c, run_js]},
  {"messages": [{"role": "user", "content": "two calls"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": "run_js", "arguments": {"code": "console.log(1+1)"}}},
                    {"type": "function", "function": {"name": "run_c", "arguments": {"source": code}}}]},
                {"role": "tool", "content": "2"}, {"role": "tool", "content": "42"}], "tools": [run_c, run_js], "effort": "low"},
  {"messages": [{"role": "user", "content": "no gen"}], "add_generation_prompt": False},
  {"messages": [{"role": "user", "content": "<tool_response>x</tool_response>"}, {"role": "user", "content": "real question"}]},
]
lines = [json.dumps(c) for c in cases]
out = subprocess.run([exe], input=("\n".join(lines) + "\n").encode(), capture_output=True, check=True).stdout.split(b"\0")
ok = 0
for i, c in enumerate(cases):
    kw = dict(messages=c["messages"], tools=c.get("tools"), add_generation_prompt=c.get("add_generation_prompt", True))
    eff = c.get("effort")
    if eff == "none": kw["enable_thinking"] = False
    elif eff: kw["reasoning_effort"] = eff
    ref = tpl.render(**kw)
    mine = out[i].decode()
    if ref == mine: ok += 1
    else:
        import difflib
        print("CASE", i, "differs")
        for l in difflib.unified_diff(ref.splitlines(), mine.splitlines(), lineterm="", n=1): print("  ", l[:200])
print(f"{ok}/{len(cases)} templates identical")
