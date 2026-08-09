# How a voice becomes an Arm-native model

This is the *why* behind the optimization, not part of it. The Arm work is in
[`../2_arm_optimization/`](../2_arm_optimization); this directory exists so the
premise — **one voice is all anyone needs** — is something you can inspect
rather than take on faith.

```
your voice  ──►  dataset  ──►  distilled VITS student  ──►  ONNX  ──►  Arm CPU
  or a teacher    validate       train (GPU, hours)         export     no GPU
     voice        FIRST                                                forever
```

The GPU builds the model **once**. After that the Arm CPU runs it, on hardware
you own, with no network and no subscription.

---

## What you can run here

```bash
python3 1_validate_dataset.py /path/to/your/dataset
```

Python 3 and the standard library. Nothing else.

This is the piece worth taking. Training a bad dataset fails **slowly** and
confusingly — often thousands of steps in, or worse, it trains happily on 40
clips and produces noise. Every check in this script is one that has actually
gone wrong on this project:

- `metadata.csv` with the wrong delimiter
- wav paths that include the folder prefix, so nothing resolves
- a mix of sample rates
- stereo files where mono is expected
- clips long enough to teach the model to stop mid-phrase

Expected layout — the same one the released Hugging Face dataset uses:

```
<dir>/metadata.csv        <wav filename>|<transcript>, one per line
<dir>/wavs/000001.wav     24 kHz mono 16-bit PCM
```

Exit 0 if usable, 1 if not, and it says *which* check failed and *why* rather
than dying with a traceback four hours into a run.

---

## What is here to read, not to run

`reference/` holds the actual source of the rest of the pipeline, so the
process is auditable:

| file | stage |
|---|---|
| `4_build_corpus.py` | assemble the prompt text |
| `5_generate_audio.py` | synthesize the corpus from a teacher voice |
| `train.py` + `config.json` | the GPL-free VITS trainer |
| `export_onnx.py` | checkpoint → the 68.5 MB ONNX graph that ships |

**These are deliberately not presented as something to execute.** Training
needs a GPU, hours of compute, PyTorch and a teacher model. Someone who starts
it expecting a two-minute demo and stalls has learned something misleading about
this project, and there is no upside to set against that. The runnable,
verifiable claims here are the thread sweep and the packages.

---

## Provenance — the part that makes cloning defensible

The obvious question about "clone any voice" is what stops misuse. The answer
ships inside both packages, and it is not a promise:

Every clip the server generates carries a **C2PA-shaped manifest** — a claim
holding labelled assertions, signed with Ed25519:

```
c2pa.actions        created, softwareAgent,
                    digitalSourceType = trainedAlgorithmicMedia
c2pa.hash.data      sha256 of the wav, size, mime
voiceyog.model      model id, version, sha256, sample rate
voiceyog.runtime    package + runtime versions, providers, threads, platform
voiceyog.request    request id, text sha256, duration, synth ms, RTF
```

The prompt text is **hashed, never stored** — provenance should establish what
produced a clip without becoming a transcript log.

Any recipient can verify it offline, with neither the model nor the private
key:

```bash
.venv/bin/python3 examples/verify_output.py <outputs>/<request-id>
```

Tamper detection is tested, not asserted (`qa/check_package.py` in the bundle):
flip one bit in the wav and `audio_matches` goes false; edit any assertion and
`signature_valid` goes false; swap the public key while keeping the key id and
it is rejected on both counts.

**What it does and does not prove.** It proves a clip came from *this install*
and has not been altered since. It does not attest to a person or an
organisation — the key is generated locally. Binding output to an identity
needs a certificate chain and a key in a keychain or HSM; that upgrade path is
documented in the bundle's `ARCHITECTURE.md`.

---

## Honest provenance of this model

- **kokoro-heart-new v3** was distilled from
  [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0) using
  its `af_heart` voice as the teacher. Kokoro is therefore a hard quality
  ceiling by construction.
- An earlier model in this project, `tarun_voice`, was trained from **a real
  person's recorded voice** and is the proof that the cloning path works
  end to end. It was produced by the **previous** version of this pipeline, not
  the one described here.
- Clone your own voice, or one you have the right to use. The teacher here is
  Apache-2.0 licensed, which is why this model can be distributed at all.
