#!/usr/bin/env bash
# ============================================================================
# pack_transfer.sh - assemble the lean transfer bundle for the MI300X instance.
#
# Bundles: driver + setup/run scripts + requirements + README,
#          the 7 flattened accent references, and the canonical word CSV.
# Does NOT bundle omnivoice or model weights (cloned/downloaded on instance).
#
# Output: dist/namedrop_accent_pack/  and  dist/namedrop_accent_pack.tar.gz
#
#   ./pack_transfer.sh
#   REF_SRC=/path/to/reference_audio ./pack_transfer.sh
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

REF_SRC="${REF_SRC:-/home/choiharam/works/projects/voice_test/transliteration/reference_audio}"
CSV_SRC="${CSV_SRC:-$REPO_ROOT/whisper_tree/whisper_audio_words_v2.csv}"

STAGE="$HERE/dist/namedrop_accent_pack"
rm -rf "$STAGE"
mkdir -p "$STAGE/reference_audio"

# accent id -> source file (relative to REF_SRC). japanese/ukrainian use the
# selected non-00 takes; the rest use 00/.
declare -A REFMAP=(
  [korean]="00/ref_korean.wav"
  [japanese]="ref_japanese__ja_regen__sent1__seed21605.wav"
  [arabic]="00/ref_arabic.wav"
  [russian]="00/ref_russian.wav"
  [ukrainian]="ref_ukrainian__whisper_female_young_adult__seed28272.wav"
  [american]="00/ref_american.wav"
  [british]="00/ref_british.wav"
)

echo "Staging references:"
for acc in "${!REFMAP[@]}"; do
  src="$REF_SRC/${REFMAP[$acc]}"
  [ -f "$src" ] || { echo "  MISSING: $src"; exit 1; }
  cp "$src" "$STAGE/reference_audio/ref_${acc}.wav"
  echo "  ref_${acc}.wav  <-  ${REFMAP[$acc]}"
done

echo "Staging word CSV:"
[ -f "$CSV_SRC" ] || { echo "  MISSING: $CSV_SRC"; exit 1; }
cp "$CSV_SRC" "$STAGE/whisper_audio_words_v2.csv"
echo "  whisper_audio_words_v2.csv  ($(wc -l <"$CSV_SRC") lines)"

echo "Staging scripts:"
for f in generate_accent_qa_mi300x.py setup_instance.sh run_accent_gen.sh \
         requirements_qa.txt README.md; do
  cp "$HERE/$f" "$STAGE/$f"
  echo "  $f"
done
chmod +x "$STAGE"/*.sh "$STAGE"/*.py

TARBALL="$HERE/dist/namedrop_accent_pack.tar.gz"
tar -C "$HERE/dist" -czf "$TARBALL" namedrop_accent_pack
echo
echo "Pack ready:"
du -sh "$STAGE" "$TARBALL" | sed 's/^/  /'
echo
echo "Send it, e.g.:"
echo "  scp $TARBALL <instance>:~/"
echo "  ssh <instance> 'tar xzf namedrop_accent_pack.tar.gz'"
