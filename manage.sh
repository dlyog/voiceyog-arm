#!/usr/bin/env bash
#
# VoiceYog on Arm — one command, from a clean machine to a voice.
#
#   bash manage.sh install     detect the machine, fetch the package, install it
#   bash manage.sh start       serve it in the background, print the URL
#   bash manage.sh stop
#   bash manage.sh status
#   bash manage.sh log [N]
#   bash manage.sh demo [text] synthesize once from the command line
#   bash manage.sh uninstall   remove the installed copy (keeps your signing key)
#
# What this is and is not. The real work lives inside the downloaded package:
# it carries the model, the runtime, 32 Python wheels for this platform and its
# own install.sh/start.sh, and it needs no network once you have it. This
# script is the orchestration around that -- figure out which package this
# machine needs, fetch and verify it, hand off, and remember where it went so
# start/stop/status/log have something to talk to.
#
# It deliberately does NOT reimplement the bundle's installer. Two installers
# that drift apart is how "works on my machine" happens, and the bundle's is
# the one a judge runs directly if they skip this script.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$HERE/.voiceyog-arm-state"

# --- output -----------------------------------------------------------------
# Deliberately long names. An earlier version of this file used $B for the bold
# escape and ALSO for the bundle path in a function local, so every heading
# printed a directory instead of turning bold.
if [ -t 1 ]; then
  C_BOLD=$'\033[1m'; C_GRN=$'\033[0;32m'; C_YEL=$'\033[0;33m'
  C_RED=$'\033[0;31m'; C_OFF=$'\033[0m'
else
  C_BOLD=""; C_GRN=""; C_YEL=""; C_RED=""; C_OFF=""
fi
say()  { printf '  %s\n' "$*"; }
ok()   { printf '  %sok%s    %s\n' "$C_GRN" "$C_OFF" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$C_YEL" "$C_OFF" "$*"; }
die()  { printf '\n  %sfail%s  %s\n\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }
head2(){ printf '\n  %s%s%s\n\n' "$C_BOLD" "$*" "$C_OFF"; }

# `start.sh --status` prints "running (pid N) on http://host:port" or
# "not running", and exits 0 either way -- so the state has to be read from the
# text. "not running" also contains the word "running", which is why the match
# is on "running (pid".
status_line() { ( cd "$1" && bash start.sh --status 2>/dev/null ); }
is_running()  { status_line "$1" | grep -q 'running (pid'; }
port_of()     { status_line "$1" | sed -n 's#.*http://[^:]*:\([0-9][0-9]*\).*#\1#p' | head -1; }

# --- which machine ----------------------------------------------------------
detect_target() {
  case "$(uname -s):$(uname -m)" in
    Darwin:arm64)  echo "apple-silicon" ;;
    Linux:aarch64) echo "dgx-spark" ;;
    Darwin:x86_64) die "Intel Macs are not supported.
        The packages ship arm64 wheels only, so pip would resolve nothing
        and fail confusingly. Apple Silicon or aarch64 Linux." ;;
    *) die "unsupported machine: $(uname -s) $(uname -m)
        This runs on macOS arm64 (M-series) or aarch64 Linux (DGX Spark)." ;;
  esac
}

espeak_hint() {
  case "$(uname -s)" in
    Darwin) echo "brew install espeak-ng" ;;
    *)      echo "sudo apt-get install espeak-ng" ;;
  esac
}

# --- prerequisites ----------------------------------------------------------
# Checked BEFORE downloading 170 MB. Finding out about a missing phonemizer
# after the download is a worse experience than finding out in two seconds.
check_prereqs() {
  local target="$1" fail=0
  head2 "Prerequisites"

  say "machine            $(uname -s) $(uname -m) -> ${target}"

  if command -v espeak-ng >/dev/null 2>&1; then
    ok "espeak-ng          $(espeak-ng --version 2>/dev/null | head -1 | awk '{print $3}')"
  else
    printf '  %sfail%s  espeak-ng is not installed\n' "$C_RED" "$C_OFF"
    say "                   $(espeak_hint)"
    say ""
    say "                   It is the phonemizer -- text becomes phonemes"
    say "                   before it becomes audio. It runs as a separate"
    say "                   process and is never linked, which is what keeps"
    say "                   its GPL terms out of the shipped runtime."
    fail=1
  fi

  local py=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      local v; v="$("$c" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
      case "$v" in 3.10|3.11|3.12) py="$c"; break ;; esac
    fi
  done
  if [ -n "$py" ]; then
    ok "python             $($py -V 2>&1 | awk '{print $2}') at $(command -v "$py")"
  else
    printf '  %sfail%s  no Python 3.10-3.12 found\n' "$C_RED" "$C_OFF"
    say "                   onnxruntime publishes no wheel for 3.13/3.14, so pip"
    say "                   would try to build from source. Not a one-command install."
    fail=1
  fi

  for t in curl unzip; do
    command -v "$t" >/dev/null 2>&1 && ok "$(printf '%-18s' "$t")present" \
      || { printf '  %sfail%s  %s is not installed\n' "$C_RED" "$C_OFF" "$t"; fail=1; }
  done

  [ "$fail" = "0" ] || die "fix the above and run this again"
  echo
}

# --- where the installed bundle lives ---------------------------------------
bundle_dir() {
  [ -f "$STATE" ] && . "$STATE" 2>/dev/null
  if [ -n "${BUNDLE:-}" ] && [ -d "$BUNDLE" ]; then echo "$BUNDLE"; return 0; fi
  # No state file, or it points at something gone. Look for an unpacked bundle.
  local d; d="$(find "$HERE/1_packages" -maxdepth 1 -type d \
                -name 'voiceyog-local-tts-*' 2>/dev/null | sort | tail -1)"
  [ -n "$d" ] && { echo "$d"; return 0; }
  return 1
}

need_bundle() {
  local d; d="$(bundle_dir)" || die "not installed yet — run:  bash manage.sh install"
  [ -x "$d/start.sh" ] || die "$d does not look like a VoiceYog package"
  echo "$d"
}

# --- commands ---------------------------------------------------------------
cmd_install() {
  local target; target="$(detect_target)"
  check_prereqs "$target"

  head2 "Package"
  local zip="voiceyog-kokoro-heart-new-v3-${target}.zip"

  if [ -f "$HERE/1_packages/$zip" ]; then
    ok "already downloaded ($zip)"
  else
    bash "$HERE/1_packages/download.sh" --target "$target" \
      || die "could not fetch the package.

        If the GitHub release is not published yet, download the zip by hand
        and drop it in 1_packages/, then run this again:
          $HERE/1_packages/$zip"
  fi

  # Unpack next to the zip so everything this script created lives in one place
  # and 'uninstall' can be honest about what it removes.
  local dir
  dir="$(unzip -Z1 "$HERE/1_packages/$zip" | head -1 | cut -d/ -f1)"
  [ -n "$dir" ] || die "could not read the archive: $HERE/1_packages/$zip"

  if [ -d "$HERE/1_packages/$dir" ]; then
    ok "already unpacked ($dir)"
  else
    say "unpacking..."
    ( cd "$HERE/1_packages" && unzip -q "$zip" ) || die "unzip failed"
    ok "unpacked $dir"
  fi

  local bd="$HERE/1_packages/$dir"
  printf 'BUNDLE="%s"\n' "$bd" > "$STATE"

  head2 "Install"
  # The bundle's own installer: verifies 102 checksums, builds a private venv,
  # installs from the vendored wheels with pip never touching the network.
  ( cd "$bd" && bash install.sh ) || die "the package installer failed (above)"

  head2 "Starting"
  cmd_start
}

cmd_start() {
  local bd; bd="$(need_bundle)" || exit 1
  if is_running "$bd"; then
    warn "already running"
    cmd_status
    return 0
  fi
  ( cd "$bd" && bash start.sh --background ) || die "could not start (see: bash manage.sh log)"
  echo
  cmd_status
}

cmd_stop() {
  local bd; bd="$(need_bundle)" || exit 1
  ( cd "$bd" && bash start.sh --stop )
}

cmd_status() {
  local bd; bd="$(bundle_dir)" || { say "not installed"; return 1; }
  if ! is_running "$bd"; then
    say "not running        (bash manage.sh start)"
    return 1
  fi
  status_line "$bd"

  # The URL and the loaded model, asked of the server rather than assumed from
  # anything this script remembers.
  local port; port="$(port_of "$bd")"
  [ -n "$port" ] || return 0
  local url="http://127.0.0.1:$port"
  local model
  model="$(curl -s --max-time 3 "$url/model" 2>/dev/null \
           | "$bd/.venv/bin/python3" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print("%s:%s  %d threads  %.1f MB" % (d["model_id"], d["model_version"],
                                          d["threads"], d["model_bytes"] / 1048576))
except Exception:
    pass' 2>/dev/null)"
  echo
  [ -n "$model" ] && say "serving            $model"
  printf '  %sopen%s  %s\n' "$C_GRN" "$C_OFF" "$url"
  echo
}

cmd_log() {
  local bd; bd="$(need_bundle)" || exit 1
  local n="${1:-60}"
  # Newest first: a bundle accumulates logs across restarts, and the
  # interesting one is always the most recent.
  local f
  f="$(find "$bd" "${VOICEYOG_LOCAL_HOME:-$HOME/.voiceyog-local-tts}" \
        -maxdepth 2 -name '*.log' 2>/dev/null \
        | while read -r p; do
            printf '%s\t%s\n' "$(date -r "$p" +%s 2>/dev/null || echo 0)" "$p"
          done | sort -rn | head -1 | cut -f2-)"
  [ -n "$f" ] || die "no log file found yet — has it been started?"
  say "$f"
  echo
  if [ -s "$f" ]; then
    tail -n "$n" "$f"
  else
    # Empty is the healthy case, not a missing file. start.sh runs uvicorn at
    # --log-level warning, so nothing is written unless something goes wrong;
    # there are no startup banners and no per-request access lines to see.
    # Printing nothing at all here just looks broken, so say what empty means.
    say "(empty — this log only receives warnings and errors, so an empty"
    say " file means the server has not complained. Use 'status' to see"
    say " whether it is up.)"
  fi
}

cmd_demo() {
  local bd; bd="$(need_bundle)" || exit 1
  local text="${*:-The Arm CPU does the majority of the work on every utterance.}"
  local port; port="$(port_of "$bd")"
  [ -n "$port" ] || die "not running — run:  bash manage.sh start"
  curl -s -X POST "http://127.0.0.1:$port/tts" \
       -H 'Content-Type: application/json' \
       -d "$("$bd/.venv/bin/python3" -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$text")" \
    | "$bd/.venv/bin/python3" -c '
import json, sys
d = json.load(sys.stdin)
m = d["measurements"]
print()
print("  %.2fs of audio in %.0f ms   RTF %.5f" % (m["audio_seconds"], m["synth_ms"], m["rtf"]))
print("  wav: %s" % d["files"]["audio"])
print()'
}

cmd_uninstall() {
  local bd; bd="$(bundle_dir)" || { say "nothing installed"; return 0; }
  ( cd "$bd" && bash start.sh --stop >/dev/null 2>&1 )
  say "removing $bd"
  rm -rf "$bd"; rm -f "$STATE"
  ok "removed"
  say ""
  say "Left alone on purpose:"
  say "  ~/.voiceyog-local-tts/keys      your signing key — deleting it would"
  say "                                  invalidate every clip you have signed"
  say "  ~/.voiceyog-local-tts/outputs   generated audio and its provenance"
  say "  1_packages/*.zip                the download, so a reinstall is offline"
}

usage() {
  cat <<EOF

  ${C_BOLD}VoiceYog on Arm${C_OFF} — single-voice TTS on an Arm CPU, no GPU

    bash manage.sh install     detect this machine, fetch the package, install, start
    bash manage.sh start       serve in the background, print the URL
    bash manage.sh stop
    bash manage.sh status      what is running, which model, how many threads
    bash manage.sh log [N]
    bash manage.sh demo [text] synthesize once from the command line
    bash manage.sh uninstall

  First time on a clean machine:

    $(espeak_hint)
    git clone https://github.com/dlyog/voiceyog-arm.git
    cd voiceyog-arm
    bash manage.sh install

EOF
}

case "${1:-}" in
  install)            cmd_install ;;
  start)              cmd_start ;;
  stop)               cmd_stop ;;
  status)             cmd_status ;;
  log)                shift; cmd_log "$@" ;;
  demo)               shift; cmd_demo "$@" ;;
  uninstall)          cmd_uninstall ;;
  ""|-h|--help|help)  usage ;;
  *)                  usage; die "unknown command: $1" ;;
esac
