# Compare the C tokenizer with HF tokenizers on a varied corpus. Run with the test venv python.
import subprocess, sys, random
tj, exe = sys.argv[1], sys.argv[2]
from tokenizers import Tokenizer
hf = Tokenizer.from_file(tj)
cases = [
    "Hello, world!", "  leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\r\n\r\nend",
    "Ciao, perché è così? Più città: Roma, Milano.", "naïve café résumé", "é composed? ñ",
    "HTTPServer XMLHttpRequest iPhone camelCaseWord SCREAMING_CASE", "I'm you're they've he'll she'd it's DON'T",
    "1234567890 3.14159 1,000,000 -42 0x1F", "数学是科学的语言。日本語のテキスト、한국어 텍스트", "emoji 😀🚀👨‍👩‍👧 done",
    "def foo(x):\n    return x ** 2  # comment\n\n\nclass Bar:\n\tpass", "a/b/c path/to/file.txt //comment\n/* c */",
    "URL https://example.com/a?b=c&d=e#frag", "    \n\n    \n", "x" * 300, "Ω≈ç√∫˜µ≤≥÷ ¡™£¢∞§¶•ªº", "क्ष Devanagari",
    "Ünïcödé Mïxed ÇÄSË ﬁ ligature ﬀ", "<think>reasoning</think> <tool_call>{\"a\":1}</tool_call>", "]~b]system\nhi[e~[",
    "mixed  double  spaces and nbsp emspace", "line sep para", "ÀÉÎÕÜ àéîõü", "'s 'S 'Re 'LL",
]
rnd = random.Random(1)
alphabet = "abcXYZ019 \n\t.,;:'\"!?()[]{}<>/\\-_=+*&^%$#@~`|éàüßçñ漢字😀́ "
for _ in range(400):
    cases.append("".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 60))))
data = b"".join(c.encode() + b"\0" for c in cases)
special = len(sys.argv) > 3
out = subprocess.run([exe, tj] + (["special"] if special else []), input=data, capture_output=True, check=True)
mine = out.stdout.decode().split("\n")
print(out.stderr.decode().strip())
bad = 0
for i, c in enumerate(cases):
    ref = hf.encode(c, add_special_tokens=False).ids
    got = [int(x) for x in mine[i].split()] if mine[i] else []
    if ref != got:
        bad += 1
        if bad <= 8:
            print("MISMATCH", repr(c[:80]))
            print("  hf  ", ref[:30], [hf.id_to_token(x) for x in ref[:12]])
            print("  mine", got[:30], [hf.id_to_token(x) for x in got[:12]])
print(f"{len(cases) - bad}/{len(cases)} match")
