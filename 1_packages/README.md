# The packages

Two self-contained zips, one per Arm target. **They are release assets, not
files in this repository** — GitHub hard-rejects anything over 100 MB in the
tree, and Git LFS bills bandwidth per download, so a repo that got any
attention would simply stop serving them.

```bash
bash download.sh                        # detects your machine, verifies sha256
bash download.sh --target dgx-spark     # or ask for one explicitly
```

| target | hardware | size |
|---|---|---|
| `apple-silicon` | macOS 13+ on M-series | 171 MB |
| `dgx-spark` | aarch64 Linux, DGX Spark GB10 | 154 MB |

Or take them straight from [Releases](../../releases/latest) and check them
against [`SHA256SUMS`](SHA256SUMS) yourself.

## What is inside

Everything needed to run, and nothing that needs fetching:

```
models/          the 68.5 MB ONNX graph + its voice config
voiceyog_local_tts/   the inference runtime (~900 lines) + vendored engine
wheels/          32 Python wheels for THIS platform
examples/        client.py, verify_output.py — stdlib only
qa/              executable checks
SHA256SUMS       102 files
install.sh  start.sh  check_system.sh  README.md  API.md  ARCHITECTURE.md
```

```bash
unzip voiceyog-kokoro-heart-new-v3-apple-silicon.zip
cd voiceyog-local-tts-kokoro-heart-new-apple-silicon-1.0.0
bash install.sh && bash start.sh
```

`install.sh` verifies all 102 checksums, creates a private virtualenv, installs
from the bundled wheels with **pip never reaching the network**, and never
touches system Python. It refuses with a stated reason — rather than
half-working — on an unsupported architecture, a macOS older than the
onnxruntime wheel supports, or a Python with no wheel available.

Transcripts of both installs from a clean extraction on real hardware are in
[`../3_evidence/`](../3_evidence).

## Why the weights are bundled rather than fetched

`package.sh` in the parent project can build a 30 MB bundle that downloads
weights from Hugging Face on first run. **This submission deliberately does not
use that mode.** The weights are on Hugging Face as the citable source of
record, but the thing you download and run has no network dependency at install
time — no proxy to negotiate, no rate limit to hit, no service to be up.

One controlled download of a checksummed artifact is a completely different
risk profile from an install that fetches from PyPI and a model hub while it
runs.

## Requirements

`espeak-ng`, and Python 3.10–3.12.

```bash
brew install espeak-ng          # macOS
sudo apt-get install espeak-ng  # Ubuntu / DGX OS
```

espeak-ng is GPL-3.0 and is invoked as a **separate process**, never linked —
running a program is not linking. The Python bindings that *would* link it
(piper-phonemize, phonemizer, espeakng-loader) are deliberately not used, which
is what keeps the shipped runtime permissively licensed throughout.

`bash check_system.sh` inside the bundle checks all of this before you start,
and is also run by `install.sh` and `start.sh`.
