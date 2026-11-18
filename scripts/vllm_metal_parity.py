"""The three rows of the blog's "Closing the gap" table, measured as its bench.py does.

Prompts and long-document construction are copied from
tarekziade/moe-expert-streaming scripts/bench.py. Greedy, thinking off, exactly
N new tokens; decode tok/s excludes the first token; the story row is the
median over the four prompts, run in order on a fresh server (cold cache first).
"""

import json
import statistics
import sys
import time

from openai import OpenAI
from transformers import AutoTokenizer

MODEL = sys.argv[1]
PORT = sys.argv[2] if len(sys.argv) > 2 else "8123"

STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."
DECODE_PROMPTS = [
    STORY,
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the causes of the French Revolution to a high school student.",
    "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'",
]

client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="x", timeout=3600)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def long_prompt(target):
    notes = [
        f"Note {i}: the harbour town kept records of ships, storms, cargo and the keepers who tended its light, "
        f"and entry {i} describes one season in detail, with names, dates and the weather."
        for i in range(1, 2000)
    ]
    text = []
    for note in notes:
        text.append(note)
        if len(tokenizer("\n".join(text)).input_ids) >= target:
            break
    return "Read these records.\n" + "\n".join(text) + "\nSummarize the history of the harbour town."


def generate(prompt, new_tokens):
    start = time.perf_counter()
    first = None
    usage = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=new_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "min_tokens": new_tokens,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter()
    end = time.perf_counter()
    ttft = first - start
    n = usage.completion_tokens
    return {
        "prompt_tokens": usage.prompt_tokens,
        "new_tokens": n,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(usage.prompt_tokens / ttft, 1),
        "decode_tok_s": round((n - 1) / (end - first), 2),
    }


report = {"story": [generate(p, 200) for p in DECODE_PROMPTS]}
report["decode_median"] = statistics.median(r["decode_tok_s"] for r in report["story"])
report["prefill_4k"] = generate(long_prompt(4000), 32)
report["long_doc_6k"] = generate(long_prompt(6000), 200)
print(json.dumps(report, indent=1))
print(
    f"RESULT decode_200={report['decode_median']:.1f} tok/s  "
    f"prefill_4k={report['prefill_4k']['prefill_tok_s']:.0f} tok/s  "
    f"decode_after_6k={report['long_doc_6k']['decode_tok_s']:.1f} tok/s"
)
