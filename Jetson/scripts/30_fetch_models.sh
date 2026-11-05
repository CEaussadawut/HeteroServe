#!/usr/bin/env bash
# Download the GGUFs named in MODELS (config/node.env) or on the command line,
# resuming partial transfers, and record sha256 so the four nodes can be proven
# to have run the same bytes.
#   ./30_fetch_models.sh                       # everything in $MODELS
#   ./30_fetch_models.sh qwen2.5-1.5b-instruct-q4_k_m
#   ./30_fetch_models.sh --list
. "$(dirname "$0")/lib/common.sh"
load_env
need_cmd curl

MANIFEST="${HS_JETSON_DIR}/config/models.tsv"

if [ "${1:-}" = "--list" ]; then
  printf '%-32s %s\n' "SHORT NAME" "FILE"
  grep -v '^#' "$MANIFEST" | grep -v '^$' | awk -F'\t' '{printf "%-32s %s\n", $1, $2}'
  exit 0
fi

WANT=( "$@" )
[ ${#WANT[@]} -eq 0 ] && read -r -a WANT <<< "${MODELS}"

SUMS="${MODEL_DIR}/SHA256SUMS"
FAILED=0

# Bytes the server says the file is, or "" when it cannot be asked. A HEAD is
# cheap and is the only truncation check available while the manifest's sha256
# column is still "-".
remote_size() {
  ( set +o pipefail
    curl -sIL --retry 2 --max-time 20 "$1" 2>/dev/null \
      | awk 'BEGIN{IGNORECASE=1} /^content-length:/{n=$2} END{gsub(/\r/,"",n); print n}' \
      || true )
}

local_size() { wc -c < "$1" 2>/dev/null | tr -d ' ' || true; }

file_sha256() { ( cd "$(dirname "$1")" && sha256sum "$(basename "$1")" | cut -d' ' -f1 ); }

for name in "${WANT[@]}"; do
  line="$( set +o pipefail; grep -v '^#' "$MANIFEST" \
    | awk -F'\t' -v n="$name" '$1==n{print; exit}' || true )"
  [ -n "$line" ] || { warn "not in manifest: ${name} (try --list)"; continue; }
  file="$(echo "$line" | cut -f2)"
  url="$(echo  "$line" | cut -f3)"
  want_sha="$(echo "$line" | cut -f4)"
  [ "$want_sha" = "-" ] && want_sha=""
  dest="${MODEL_DIR}/${file}"
  part="${dest}.part"

  step "$name"

  # ---- is what is on disk actually complete? -------------------------------
  # `-s` (non-empty) was the old test, and it is true of a half-downloaded file.
  # That made the resume path below unreachable and let a truncated GGUF be
  # hashed into SHA256SUMS as if it were ground truth.
  complete=0
  if [ -f "$dest" ]; then
    if [ -n "$want_sha" ]; then
      have="$(file_sha256 "$dest")"
      if [ "$have" = "$want_sha" ]; then
        complete=1
        ok "already present and verified against the manifest"
      else
        warn "sha256 MISMATCH -- on disk ${have:0:16}..., manifest ${want_sha:0:16}..."
        warn "moving it aside to ${file}.bad and re-downloading"
        mv -f "$dest" "${dest}.bad"
      fi
    else
      rsize="$(remote_size "$url")"
      lsize="$(local_size "$dest")"
      if [ -n "$rsize" ] && [ -n "$lsize" ] && [ "$rsize" = "$lsize" ]; then
        complete=1
        ok "already present, size matches the server ($(du -h "$dest" | cut -f1))"
      elif [ -n "$rsize" ] && [ -n "$lsize" ]; then
        warn "TRUNCATED: ${lsize} bytes on disk, server says ${rsize} -- resuming"
        mv -f "$dest" "$part"
      else
        # No network to ask. Say so rather than assuming the file is good.
        complete=1
        warn "present ($(du -h "$dest" | cut -f1)) but UNVERIFIED: no manifest sha256"
        warn "and the server could not be reached for a size. Re-run online."
      fi
    fi
  fi

  # ---- download into a sidecar, promote only on success ---------------------
  if [ "$complete" -eq 0 ]; then
    info "$url"
    # -C - resumes; SD cards and Thai home links both drop mid-download. The
    # sidecar is the point: a partial transfer is never left at the real path,
    # so it can never be mistaken for a complete file on the next run.
    if curl -L --fail --retry 5 --retry-delay 5 -C - -o "$part" "$url"; then
      mv -f "$part" "$dest"
    else
      rc=$?
      # 33 = server refused a byte range, i.e. the sidecar is already whole.
      if [ "$rc" -eq 33 ] && [ -s "$part" ]; then
        mv -f "$part" "$dest"
      else
        # warn, not die: one dead URL must not silently skip every model after it
        warn "download failed (curl ${rc}): ${name} -- partial kept at ${file}.part"
        FAILED=$((FAILED + 1))
        continue
      fi
    fi
  fi

  # ---- verify against the manifest, then record ----------------------------
  have="$(file_sha256 "$dest")"
  if [ -n "$want_sha" ] && [ "$have" != "$want_sha" ]; then
    warn "sha256 MISMATCH after download: ${have:0:16}... != ${want_sha:0:16}..."
    mv -f "$dest" "${dest}.bad"
    FAILED=$((FAILED + 1))
    continue
  fi

  recorded="$( set +o pipefail; grep " ${file}\$" "$SUMS" 2>/dev/null | cut -d' ' -f1 || true )"
  if [ -z "$recorded" ]; then
    printf '%s  %s\n' "$have" "$file" >> "$SUMS"
    if [ -n "$want_sha" ]; then
      ok "recorded sha256 (verified against the manifest)"
    else
      warn "recorded sha256, but NOTHING verified it -- put it in column 4 of"
      warn "config/models.tsv so the next node can be checked against it"
    fi
  elif [ "$recorded" != "$have" ]; then
    # Re-verify rather than skip. The old code skipped whenever a line existed,
    # so a file that changed underneath kept its stale, passing entry forever.
    warn "sha256 CHANGED since it was recorded:"
    warn "  SHA256SUMS says ${recorded:0:16}..., the file is now ${have:0:16}..."
    FAILED=$((FAILED + 1))
    continue
  else
    ok "sha256 matches the recorded value"
  fi
  grep " ${file}\$" "$SUMS"
done

[ "$FAILED" -eq 0 ] || warn "${FAILED} model(s) failed -- see above; the rest were processed"

step "Model directory"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || warn "no models present"
df -h "$MODEL_DIR" | tail -1
echo
echo "Verify anytime with:  (cd ${MODEL_DIR} && sha256sum -c SHA256SUMS)"
echo
echo "NOTE: that only proves the files have not changed since they were recorded."
echo "It proves they are the RIGHT bytes only for models whose 4th column in"
echo "config/models.tsv holds a real sha256 rather than \"-\"."

# Exit non-zero when a model genuinely failed, so `make models` cannot report
# success over a missing or corrupt GGUF -- the same trap that let `make facts`
# pass while writing a zero-byte facts file.
[ "$FAILED" -eq 0 ] || exit 1
