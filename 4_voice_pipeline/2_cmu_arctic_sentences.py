"""
Parses the CMU ARCTIC prompt list (reference_data/cmuarctic.data) --
1,132 phonetically balanced US English sentences, selected from
out-of-copyright Project Gutenberg prose by CMU's Language Technologies
Institute specifically for single-speaker TTS synthesis research.
Source: http://www.festvox.org/cmu_arctic/cmuarctic.data

File format per line: ( arctic_a0001 "Some sentence text." )
"""
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "reference_data" / "cmuarctic.data"

_LINE_RE = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')


def cmu_arctic_sentences() -> list[str]:
    sentences = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            text = m.group(2).replace('\\"', '"').strip()
            if text:
                sentences.append(text)
    return sentences


if __name__ == "__main__":
    s = cmu_arctic_sentences()
    print(f"{len(s)} CMU ARCTIC sentences parsed")
    for x in s[:5]:
        print(" -", x)
