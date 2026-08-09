#!/usr/bin/env bash
# Fetch the package for this machine from the GitHub release, and verify it.
#
#   bash 1_packages/download.sh              # detect this machine
#   bash 1_packages/download.sh --target apple-silicon
#   bash 1_packages/download.sh --target dgx-spark
#
# The zips are release assets, not repository files: GitHub hard-rejects any
# file over 100 MB in the tree, and Git LFS bills bandwidth per download, so a
# popular repo would simply stop serving them. Release assets have neither
# limit.
#
# The checksum check is the point of this script. A truncated download is
# silent -- you get a zip that unzips and a model that loads and produces
# noise -- so the SHA256SUMS beside this file is compared before you are told
# it worked.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="dlyog/voiceyog-arm"
TAG="${VOICEYOG_TAG:-v1.0}"
TARGET=""

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:?--target needs apple-silicon or dgx-spark}"; shift 2 ;;
    --tag)    TAG="${2:?--tag needs a release tag}"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

die() { printf '  fail  %s\n' "$*" >&2; exit 1; }
ok()  { printf '  ok    %s\n' "$*"; }
say() { printf '  %s\n' "$*"; }

# --- which package ----------------------------------------------------------
if [ -z "$TARGET" ]; then
  arch="$(uname -m)"
  case "$(uname -s):$arch" in
    Darwin:arm64)   TARGET="apple-silicon" ;;
    Linux:aarch64)  TARGET="dgx-spark" ;;
    Darwin:x86_64)  die "Intel Macs are not supported: the packages ship arm64 wheels only." ;;
    *)              die "unsupported machine: $(uname -s) $arch (need macOS arm64 or aarch64 Linux)" ;;
  esac
  say "detected $(uname -s) $arch -> $TARGET"
fi

ZIP="voiceyog-kokoro-heart-new-v3-${TARGET}.zip"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ZIP}"

grep -q " ${ZIP}\$" "$HERE/SHA256SUMS" \
  || die "no checksum recorded for $ZIP -- refusing to download something unverifiable"

# --- fetch ------------------------------------------------------------------
cd "$HERE"
if [ -f "$ZIP" ]; then
  say "$ZIP is already here, skipping the download"
else
  say "downloading $ZIP (~170 MB)"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$ZIP.part" "$URL" || die "download failed: $URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$ZIP.part" "$URL" || die "download failed: $URL"
  else
    die "neither curl nor wget is installed"
  fi
  # Rename only after a complete transfer, so an interrupted download can
  # never be mistaken for a finished one on the next run.
  mv "$ZIP.part" "$ZIP"
fi

# --- verify -----------------------------------------------------------------
if command -v shasum >/dev/null 2>&1; then
  SUM="$(shasum -a 256 "$ZIP" | cut -d' ' -f1)"
elif command -v sha256sum >/dev/null 2>&1; then
  SUM="$(sha256sum "$ZIP" | cut -d' ' -f1)"
else
  die "no shasum or sha256sum available to verify the download"
fi
WANT="$(grep " ${ZIP}\$" "$HERE/SHA256SUMS" | cut -d' ' -f1)"

if [ "$SUM" != "$WANT" ]; then
  rm -f "$ZIP"
  die "checksum mismatch -- deleted the file.
        expected $WANT
        got      $SUM"
fi
ok "sha256 verified"

echo
say "next:"
say "    unzip $ZIP"
say "    cd voiceyog-local-tts-kokoro-heart-new-*"
say "    bash install.sh && bash start.sh"
echo
say "install.sh verifies 102 more checksums, installs from the 32 wheels in the"
say "bundle, and touches the network at no point."
echo
