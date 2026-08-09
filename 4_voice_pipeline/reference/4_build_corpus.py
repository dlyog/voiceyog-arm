"""
Builds the training-text manifest from every available source.

    python3 1_SyntheticAudioDataset/4_build_corpus.py [target_sentences]

Sources, in priority order:

  1_word_bank.py            fixed sentences -- pangrams, numbers, contractions,
                            punctuation. Deliberately covers the cases a
                            phoneme-level model gets wrong.
  2_cmu_arctic_sentences.py 1,132 phonetically balanced public-domain prompts
                            from CMU's Language Technologies Institute.
  3_llm_sentences.py        OPTIONAL topical variety, only if LLM_BASE_URL and
                            LLM_MODEL are set.

The first two need no network and no external service, so this produces a
usable corpus on any machine. The LLM step is additive.

Output: output/corpus.jsonl  ({id, text, source})
"""
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Module names begin with a digit, which is not a valid Python identifier, so
# they cannot be written as `import 1_word_bank`. import_module takes a string
# and loads them fine -- the numeric prefix is for humans reading the folder.
word_bank = importlib.import_module("1_word_bank")
cmu = importlib.import_module("2_cmu_arctic_sentences")
llm = importlib.import_module("3_llm_sentences")

DEFAULT_TARGET = 4500          # ~3 h of audio at the observed ~2.6 s/sentence
OUTPUT_PATH = HERE / "output" / "corpus.jsonl"
LLM_CACHE_PATH = HERE / "output" / "llm_sentences_cache.jsonl"


def build(target: int = DEFAULT_TARGET) -> Path:
    rows = [{"text": s, "source": "word_bank"} for s in word_bank.fixed_sentences()]
    rows += [{"text": s, "source": "cmu_arctic"} for s in cmu.cmu_arctic_sentences()]
    offline = len(rows)
    print(f"offline sources: {offline} sentences "
          f"({len(word_bank.fixed_sentences())} word bank + "
          f"{len(cmu.cmu_arctic_sentences())} CMU ARCTIC)")

    shortfall = target - offline
    if shortfall <= 0:
        print(f"target {target} already met without the LLM step")
    elif not llm.available():
        print(f"note: {shortfall} short of the {target} target, and no LLM is "
              f"configured.\n      Set LLM_BASE_URL and LLM_MODEL to add topical "
              f"variety, or continue\n      with {offline} sentences -- enough for "
              f"roughly {offline * 2.6 / 3600:.1f} h of audio.")
    else:
        print(f"LLM configured; generating {shortfall} more sentences ...")
        rows += [{"text": s, "source": "llm"} for s in
                 llm.generate_sentences(target_count=shortfall,
                                        cache_path=LLM_CACHE_PATH)]

    seen, deduped = set(), []
    for r in rows:
        key = r["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for i, r in enumerate(deduped):
            f.write(json.dumps({"id": i, "text": r["text"], "source": r["source"]}) + "\n")

    wc = [len(r["text"].split()) for r in deduped]
    by_source = {}
    for r in deduped:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"\ncorpus: {len(deduped)} sentences "
          f"({len(rows) - len(deduped)} duplicates dropped)")
    for k, v in by_source.items():
        print(f"  {k:12} {v}")
    print(f"words per sentence: min={min(wc)} max={max(wc)} avg={sum(wc)/len(wc):.1f}")
    print(f"written to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET)
