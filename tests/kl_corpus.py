#!/usr/bin/env python3
"""Token corpus for measuring a quantization recipe's KL against the exact engine: a multi-turn chat with
Italian prose, an English explanation, C and JavaScript code with tests. Written for this project.

    tvenv/bin/python -I tests/kl_corpus.py MODEL_DIR OUT.txt   -> comma-separated token ids
"""
import sys

from transformers import AutoTokenizer

IT = """Le reti neurali ricorrenti hanno avuto per anni un problema serio: per ricordare qualcosa dovevano
comprimere tutto il passato in un vettore di dimensione fissa, e l'addestramento faticava a propagare il
gradiente su sequenze lunghe. L'attenzione ha risolto il problema in modo brutale, confrontando ogni token con
tutti quelli precedenti, ma il costo cresce con il quadrato della lunghezza. I modelli ibridi di oggi cercano
un compromesso: alcuni strati usano una memoria ricorrente compatta, aggiornata con la regola delta, mentre
pochi strati di attenzione sparsa recuperano i dettagli quando servono. Il risultato è un modello che legge
documenti lunghi senza che la memoria esploda, e che resta preciso quando deve citare una riga esatta."""

EN = """A mixture-of-experts layer replaces one large feed-forward block with many small ones. A router scores
the experts for every token and only the best few run, so the compute per token stays small while the total
number of parameters grows. The catch is memory: every expert must be stored somewhere, and the set of experts
a token needs is only known once the router has seen it. On a laptop the experts live partly on the SSD, and the
engine's job is to keep the right ones in RAM."""

C_CODE = r'''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Splits s on any character in delims; returns a NULL-terminated array of malloc'ed tokens. */
char **split(const char *s, const char *delims, size_t *count) {
    size_t cap = 8, n = 0;
    char **out = malloc(cap * sizeof *out);
    if (!out) return NULL;
    while (*s) {
        s += strspn(s, delims);
        if (!*s) break;
        size_t len = strcspn(s, delims);
        if (n + 1 == cap) {
            char **t = realloc(out, 2 * cap * sizeof *out);
            if (!t) goto fail;
            out = t;
            cap *= 2;
        }
        out[n] = malloc(len + 1);
        if (!out[n]) goto fail;
        memcpy(out[n], s, len);
        out[n++][len] = '\0';
        s += len;
    }
    out[n] = NULL;
    if (count) *count = n;
    return out;
fail:
    for (size_t i = 0; i < n; i++) free(out[i]);
    free(out);
    return NULL;
}

int main(void) {
    size_t n;
    char **w = split("  uno, due;tre  ", " ,;", &n);
    if (!w || n != 3 || strcmp(w[0], "uno") || strcmp(w[2], "tre")) { puts("FAIL"); return 1; }
    for (size_t i = 0; i < n; i++) free(w[i]);
    free(w);
    puts("ok");
    return 0;
}
'''

JS_CODE = r'''// Groups items by a key function, keeping the first-seen order of the keys.
function groupBy(items, keyFn) {
  const groups = new Map();
  for (const item of items) {
    const key = keyFn(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  return groups;
}

const assert = require("assert");
const people = [{ name: "Anna", city: "Roma" }, { name: "Bruno", city: "Milano" }, { name: "Carla", city: "Roma" }];
const byCity = groupBy(people, (p) => p.city);
assert.deepStrictEqual([...byCity.keys()], ["Roma", "Milano"]);
assert.strictEqual(byCity.get("Roma").length, 2);
assert.deepStrictEqual(groupBy([], (x) => x), new Map());
console.log("ok");
'''

messages = [
    {"role": "user", "content": "Spiegami in parole semplici perché i modelli ibridi con memoria ricorrente sono utili."},
    {"role": "assistant", "content": IT},
    {"role": "user", "content": "Now explain mixture-of-experts layers in English, briefly."},
    {"role": "assistant", "content": EN},
    {"role": "user", "content": "Scrivi una funzione C che divide una stringa in token, con un piccolo test."},
    {"role": "assistant", "content": "Ecco una versione che gestisce la memoria con attenzione:\n\n```c\n" + C_CODE + "```\n\n"
                                     "La funzione libera tutto in caso di errore, e il test controlla i separatori ripetuti."},
    {"role": "user", "content": "E in JavaScript una groupBy che mantiene l'ordine delle chiavi, con dei test."},
    {"role": "assistant", "content": "```javascript\n" + JS_CODE + "```\n\nUso una `Map` perché conserva l'ordine di inserimento "
                                     "delle chiavi e accetta chiavi di qualunque tipo."},
]

tok = AutoTokenizer.from_pretrained(sys.argv[1])
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
ids = tok(text, add_special_tokens=False)["input_ids"]
open(sys.argv[2], "w").write(",".join(map(str, ids)))
print(len(ids), "tokens")
