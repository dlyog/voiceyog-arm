# VoiceYog — demo video

## The one to submit

**`voiceyog_arm_demo_mac_and_dgx.mp4` — 2 min 39 s**

Two machines side by side, one narration. **Left: Apple M1 Max. Right: NVIDIA
DGX Spark GB10.** Same repository, same commands, same model — cloned fresh and
installed from scratch on both.

**The narration is synthesized by kokoro-heart-new v3** — the 68.5 MB model the
video is about — running on the Arm CPU with no GPU and no network. The voice
explaining the optimization is the artifact being optimized.

`voiceyog_arm_full_demo.mp4` (3 min 14 s) is the earlier Mac-only cut, kept as a
fallback.

---

## Why side by side is the whole argument

Scene 4 is the shot that does the work. Both panes show the same model, same
version, same 68.5 MB — and then:

```
Mac         threads: 8        DGX Spark   threads: 9
```

Nothing was configured. Scene 11 shows where those numbers came from: the Mac
reports 8 performance and 2 efficiency cores; the DGX Spark reports ten
Cortex-X925 beside ten Cortex-A725 **and warns that the fast cores are not
contiguous**. Scene 12 shows what the wrong number costs.

One graph, one binary, retuned per Arm part. In one frame.

---

## Scene-by-scene

Text below is exactly what was synthesized, so it matches the audio word for
word. Files are in `side_by_side/scenes/`, voice-over wavs in
`side_by_side/voiceover/`.

| # | scene | length | what it is there to show |
|---|---|---|---|
| 1 | `01_install_espeak` | 17.2 s | framing — the voice you hear *is* the model |
| 2 | `02_clone_repo` | 9.8 s | the repo is small and code-only |
| 3 | `03_manage_install` | 17.2 s | **reproducibility** — fresh environment, offline |
| 4 | `04_open_browser` | 13.6 s | **the 8-vs-9 reveal** — same model, different silicon |
| 5 | `05_browser_synthesize` | 12.2 s | **the Arm CPU is the deployment target** |
| 6 | `06_browser_provenance` | 14.7 s | responsible cloning |
| 7 | `07_browser_tamper` | 7.0 s | the signature fails when it should |
| 8 | `08_manage_status` | 6.4 s | 8 threads vs 9, from the CLI |
| 9 | `09_manage_demo` | 6.0 s | synthesis without the browser |
| 10 | `10_verify_claims` | 13.5 s | **anyone can check the numbers** |
| 11 | `11_core_topology` | 18.9 s | **why this is specifically an Arm optimization** |
| 12 | `12_thread_sweep` | 22.4 s | **what changed, and what it is worth** |

### 1 — `01_install_espeak`
> Everything you hear is spoken by the model this video is about. Sixty eight
> megabytes, on an Arm C P U, no GPU and no network. Left, an Apple M one Max.
> Right, an NVIDIA DGX Spark. Same repository, same commands, both machines.

### 2 — `02_clone_repo`
> Clone the repository. Code only, under six hundred kilobytes. The model
> arrives separately, as a package built for that platform.

### 3 — `03_manage_install`
> One command does the rest. It detects the machine, fetches the right package,
> verifies its checksum and installs. One hundred and two more checksums, thirty
> two wheels carried inside the archive, and pip never touches the network.
> Fresh environments are where installers fail.

### 4 — `04_open_browser`  ← **the money shot**
> Watch the difference. Same model, same version, same sixty eight and a half
> megabytes. But the Mac loads eight threads and the DGX Spark loads nine.
> Nothing was configured. That number is the optimization.

### 5 — `05_browser_synthesize`
> Both synthesize on the C P U alone, on a box with a Blackwell GPU sitting idle
> beside it. On DGX Spark the Arm side is not a host babysitting CUDA. It is the
> deployment target.

### 6 — `06_browser_provenance`
> Every clip carries a signed record of what produced it. Model, checksum,
> runtime, platform. Verifiable offline, without the model and without the
> private key. If a model can become someone's voice, that record is what makes
> it accountable.

### 7 — `07_browser_tamper`
> And it fails when it should. Flip one bit and the check goes false. Tested,
> not asserted.

### 8 — `08_manage_status`
> The same state from the command line. Eight threads on the Mac. Nine on the
> DGX Spark.

### 9 — `09_manage_demo`
> Synthesis from the command line, timed by the server rather than by us.

### 10 — `10_verify_claims`
> Run this first. Every number in the write up is emitted by a script into a
> J SON file, and this checker fails if the two disagree. Thirty three claims,
> both machines, no dependencies, no network.

### 11 — `11_core_topology`
> Here is what is being optimized. Both are asymmetric Arm parts. This reads the
> core layout from the hardware registers rather than guessing from a core
> count. Eight performance cores on the Mac. On DGX Spark, ten Cortex X nine two
> five beside ten Cortex A seven two five, and the fast cores are not next to
> each other.

### 12 — `12_thread_sweep`
> And this proves it on your machine in a minute. An operator finishes when its
> slowest thread finishes, so one thread on a slower core makes every other
> thread wait. Handing O N N X Runtime every core is one point three nine times
> slower on the DGX Spark, and two point two one times slower on the Mac. Same
> graph, same binary, retuned per Arm part. Run it yourself.

---

## Shorter cuts

**90 seconds:** scenes 3, 4, 11, 12 — install, the 8-vs-9 reveal, why, what it's
worth.
**45 seconds:** scenes 4 and 12.

---

## How it was made, and what is real

| script | role |
|---|---|
| `scripts/record_cli.js` / `record_cli_dgx.js` | render captured terminal output as an HTML terminal, record with Playwright |
| `scripts/record_webapp.js` | drive the real web app in headless Chromium and record it |
| `scripts/make_header.js` | render the machine-label strip as a PNG |
| `scripts/build_sidebyside.py` | synthesize each narration, stack the two panes, concatenate |
| `scripts/captured_output/` · `captured_output_dgx/` | the raw terminal output from each machine |

**Every line of terminal output is real** — captured by running each command on
that machine. The DGX pane is a genuine `ssh dlyog@dgx1` session in
`~/vy_demo/`, cloned from GitHub and installed from scratch. Its browser pane is
the DGX's own server, reached over an SSH tunnel, so the audio in those frames
was synthesized on the DGX's Arm CPU.

**No screen recording was used anywhere.** Playwright captures the browser's own
rendering, which is why the terminal is an HTML page — macOS Screen Recording
permission cannot be granted from a shell, and this needs no permission at all.

**Video is sped up to fit the narration; audio is never touched.** The browser
scenes run at 3× because they were recorded with generous waits. Time-compressed
speech sounds wrong immediately, and the voice is the point.

Three things worth knowing if anyone asks:

- The espeak-ng step shows "already installed" on both machines, because it was.
  On a clean machine it downloads.
- The packages were placed in `1_packages/` by hand because the GitHub release
  is not published yet; once it is, `manage.sh install` downloads them itself.
- Scene 6's shot initially showed only the top of the provenance card. It now
  scrolls to the verdict — the signature result, which is the thing the card is
  evidence for.
