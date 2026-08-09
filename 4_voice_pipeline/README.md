# Build your own voice

These are the scripts that made the model in this repository, and they are here
so you can make a different one. Two things they are for:

1. **An AI voice.** Pick any Kokoro voice, distil it into a 68.5 MB model that
   runs on a CPU.
2. **Your own voice.** Give a clone service one recording of yourself, and the
   same pipeline produces *your* voice as a 68.5 MB model that runs on a CPU,
   offline, for as long as you keep the file.

Same scripts either way. Only step 5 changes.

```
1_word_bank.py            ┐
2_cmu_arctic_sentences.py ├─ 4_build_corpus.py  → the sentences to speak
3_llm_sentences.py        ┘

5_generate_dataset.py     → a teacher speaks them   (the only step that differs)
6_validate_dataset.py     → check it BEFORE training
7_train.py                → train the student on the GPU
8_export_onnx.py          → 68.5 MB ONNX that runs on any Arm CPU
```

The GPU is used **once**, to build the model. After that it is never needed
again.

---

## The whole thing, run end to end

Every command below was run on an NVIDIA DGX Spark GB10 before being written
here, and the output is what it actually printed.

### 1 · Build the prompt corpus

```bash
python3 4_build_corpus.py
```
```
offline sources: 1194 sentences (62 word bank + 1132 CMU ARCTIC)
corpus: 1194 sentences (0 duplicates dropped)
words per sentence: min=1 max=13 avg=8.8
written to output/corpus.jsonl
```

No network. The word bank covers what phoneme models get wrong — numbers,
contractions, punctuation — and CMU ARCTIC supplies 1,132 phonetically
balanced public-domain prompts.

### 2 · Have a teacher speak it

**An AI voice** — Kokoro-82M, in-process:

```bash
python3 5_generate_dataset.py --teacher kokoro --voice af_heart --hours 3.0
```

**Your own voice** — a clone service, conditioned on one recording of you:

```bash
python3 5_generate_dataset.py --teacher clone --ref my-voice.wav \
        --api http://127.0.0.1:8005 --hours 3.0
```
```
teacher   clone via http://127.0.0.1:8005 from my-voice.wav
corpus    1194 sentences
2 clips · 0.00 h (2 new, 0 already there, 0 rejected)
output/clone/metadata.csv
```

**Qwen3-TTS** is a free model that provides exactly this and is supported on
DGX Spark. Install it and follow the [Qwen TTS guide on Hugging
Face](https://huggingface.co/Qwen); this pipeline only needs a URL that accepts
a reference wav and returns spoken text. Your recording is sent only to the URL
you pass, which is expected to be a service you run.

Each teacher writes to its own directory (`output/kokoro/`, `output/clone/`) so
a quick `--limit 5` test can never overwrite a real dataset. Resumable: clips
already on disk are skipped.

### 3 · Validate before you train

```bash
python3 6_validate_dataset.py output/kokoro
```
```
rows 4104 · wavs_on_disk 4104 · missing 0
sample_rate 24000 · est_hours 3.057 · longest_sec 5.2
OK — 4104 clips, ~3.057 h, 24000 Hz. Ready to train.
```

Worth the ten seconds. A bad dataset fails *slowly* — often thousands of steps
in, or worse, it trains happily on 40 clips and produces noise. Every check
here is one that has actually gone wrong: wrong delimiter, path prefixes in
`metadata.csv`, mixed sample rates, stereo files, clips long enough to teach
the model to stop mid-phrase.

### 4 · Train

```bash
python3 7_train.py --config config.json
```
```
dataset: 4104 clips
128 batches/epoch at batch_size=32
  generator             25.74 M
    of which vocoder     3.76 M   (ResBlock1)
  MPD                   46.75 M
  MRD                    0.28 M
  TOTAL trainable       72.77 M
epoch 0 step 50  mel 65.8324  kl 12.8677  gen 3.5651  disc 5.2028  0.96 it/s
[epoch 0] mel=54.1921 kl=6.3126 (125s)
saved runs/checkpoints/epoch0000_step128_mel54.1921.pt
```

**This needs a GPU and it needs hours.** About 125 s per epoch on a GB10 with
3 h of audio; a usable voice takes a few hundred epochs. It checkpoints every
epoch and resumes with `--resume`, so it survives being interrupted.

Trained **from scratch** — this is not a fine-tune of Kokoro. It is a smaller
VITS/HiFi-GAN model learning from Kokoro's audio, which is why the result can
be a tenth of the size rather than a compressed copy.

### 5 · Export

```bash
python3 8_export_onnx.py --checkpoint runs/checkpoints/epoch0000_step128_mel54.1921.pt \
                         --output-file my_voice.onnx
```
```
checkpoint: epoch 0, step 128, train_mel 54.1921
exported: my_voice.onnx  (68.5 MB)
config:   my_voice.onnx.json
smoke test: (1, 1, 27904) -> 1.16s of audio at 24000 Hz
```

That `.onnx` and its `.json` are what the packages in
[`../1_packages/`](../1_packages) serve. Drop them into a bundle's `models/`
directory and it will run your voice instead.

---

## What you need

| | |
|---|---|
| **GPU** | for step 4 only. A DGX Spark GB10 is what this was built and tested on. |
| **espeak-ng** | phonemisation, at training and inference |
| **PyTorch** | training and export |
| `kokoro` | only for `--teacher kokoro` |
| a clone service | only for `--teacher clone` — Qwen3-TTS on DGX Spark |
| **Nothing** | for step 3. The validator is standard library only. |

The `tts/` directory beside these scripts is the model code the trainer needs,
vendored so the pipeline is self-contained.

---

## Two honest notes

**Kokoro is a ceiling, not a floor.** A student trained on a teacher's audio
cannot exceed the teacher. What it can do is be 4.8× smaller and run without a
GPU, which is the trade this whole project is about.

**Clone your own voice, or one you have the right to use.** The provenance
signing built into every package exists for exactly this reason: each clip
carries a signed record naming the model that made it, verifiable offline by
anyone who receives it.
