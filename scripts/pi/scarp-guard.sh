#!/usr/bin/env bash
# SCARP regression guard: run the campaign's baseline matrix and diff it
# against a recorded baseline (tok/s + generated-token-stream identity).
#
# Every SCARP phase attaches a guard run. Phase G0 records the baseline; every
# later phase compares against it. See docs/bench/scarp_phase0_baseline.md.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scarp-guard.sh --out DIR [options]
       scarp-guard.sh --out DIR --baseline BASELINE_DIR [options]

Runs the SCARP baseline matrix (five models x two canonical prompts, greedy,
NANOCAMELID_TRACE=1) and writes one log per run plus a summary TSV and a
`json:` summary line. With --baseline it additionally diffs the new run
against a recorded one and exits non-zero on a regression.

Options:
  --out DIR             Directory for this run's logs and summary (required)
  --baseline DIR        Compare against a previously recorded --out directory
  --tokens N            Tokens to generate per run, default 64
  --rows LIST           Comma-separated row ids, default all known rows
  --prompts LIST        Comma-separated prompt ids (short,long), default both
  --label NAME          Free-text label recorded in the summary
  --repeats N           Runs per (row,prompt) cell, default 1
  --tolerance PCT       Allowed tok/s drop vs baseline, default 3.0
  --allow-text-drift    Do not fail when the generated token stream changes
                        (use only for a phase whose bar is token-parity, not
                        byte-identity -- and say so in the receipt)
  --env "K=V K=V"       Extra environment applied to every run
  --list                Print the known rows and exit
  --dry-run             Print the resolved plan without running anything

Env:
  NANOCAMELID_WORKSPACE   Pi workspace, default /mnt/nanocamelid
  NANOCAMELID_BIN         Binary path, default $CARGO_TARGET_DIR/release/nanocamelid
  CARGO_TARGET_DIR        Cargo output dir, default /mnt/nanocamelid/target

Rows are (id, gguf, head_kind) triples. head_kind is the SCARP-relevant
property: `tied` models have no output.weight tensor and take the tied LM head
path; `untied` models ship an output.weight and are the campaign's regression
witnesses, not its beneficiaries.
USAGE
}

# row id | gguf basename | expected head kind
ROW_TABLE=(
  "llama32-1b-q4_0|Llama-3.2-1B-Instruct-Q4_0.gguf|tied"
  "llama32-1b-q8_0|Llama-3.2-1B-Instruct-Q8_0.gguf|tied"
  "gemma3-1b-it-q4_0|gemma-3-1b-it-q4_0.gguf|tied"
  "qwen3-0.6b-q8_0|qwen3-0.6b-q8_0.gguf|tied"
  "llama3-8b-q4_k_m|Meta-Llama-3-8B-Instruct-Q4_K_M.gguf|untied"
)

PROMPT_SHORT='Explain in one sentence why the sky appears blue during the day.'
PROMPT_LONG='You are helping design a small weather station for a remote farm. The station
must run from a solar panel and a battery, survive cold winters, and report
temperature, humidity, wind speed, and rainfall once every ten minutes over a
low-bandwidth radio link. The farmer cares most about frost warnings for the
orchard, which need to be timely and reliable even when the network is flaky.
Describe a sensible hardware and software architecture for this station.
Explain which sensors you would pick, how you would manage power so the
battery lasts through a week of cloudy days, how you would buffer and resend
readings when the radio link drops, and what simple on-device rule you would
use to raise a frost alarm without waiting for the server.'

OUT_DIR=""
BASELINE_DIR=""
TOKENS=64
ROWS_RAW=""
PROMPTS_RAW="short,long"
LABEL=""
REPEATS=1
TOLERANCE="3.0"
ALLOW_TEXT_DRIFT=0
EXTRA_ENV=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --list)
      for entry in "${ROW_TABLE[@]}"; do
        IFS='|' read -r id gguf kind <<<"$entry"
        printf '%-20s %-45s %s\n' "$id" "$gguf" "$kind"
      done
      exit 0
      ;;
    --out)
      OUT_DIR="${2:?--out needs a directory}"
      shift 2
      ;;
    --baseline)
      BASELINE_DIR="${2:?--baseline needs a directory}"
      shift 2
      ;;
    --tokens)
      TOKENS="${2:?--tokens needs a value}"
      shift 2
      ;;
    --rows)
      ROWS_RAW="${2:?--rows needs a value}"
      shift 2
      ;;
    --prompts)
      PROMPTS_RAW="${2:?--prompts needs a value}"
      shift 2
      ;;
    --label)
      LABEL="${2:?--label needs a value}"
      shift 2
      ;;
    --repeats)
      REPEATS="${2:?--repeats needs a value}"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="${2:?--tolerance needs a value}"
      shift 2
      ;;
    --allow-text-drift)
      ALLOW_TEXT_DRIFT=1
      shift
      ;;
    --env)
      EXTRA_ENV="${2:?--env needs a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  echo "--out is required" >&2
  usage >&2
  exit 2
fi
if [[ ! "$TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--tokens must be a positive integer: $TOKENS" >&2
  exit 2
fi
if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeats must be a positive integer: $REPEATS" >&2
  exit 2
fi

WORKSPACE="${NANOCAMELID_WORKSPACE:-/mnt/nanocamelid}"
TARGET_DIR="${CARGO_TARGET_DIR:-${NANOCAMELID_TARGET_DIR:-/mnt/nanocamelid/target}}"
BINARY="${NANOCAMELID_BIN:-$TARGET_DIR/release/nanocamelid}"

SELECTED_ROWS=()
if [[ -z "$ROWS_RAW" ]]; then
  for entry in "${ROW_TABLE[@]}"; do
    SELECTED_ROWS+=("$entry")
  done
else
  for want in ${ROWS_RAW//,/ }; do
    found=""
    for entry in "${ROW_TABLE[@]}"; do
      IFS='|' read -r id _ _ <<<"$entry"
      if [[ "$id" == "$want" ]]; then
        found="$entry"
        break
      fi
    done
    if [[ -z "$found" ]]; then
      echo "Unknown row id: $want (see --list)" >&2
      exit 2
    fi
    SELECTED_ROWS+=("$found")
  done
fi

SELECTED_PROMPTS=()
for want in ${PROMPTS_RAW//,/ }; do
  case "$want" in
    short | long) SELECTED_PROMPTS+=("$want") ;;
    *)
      echo "Unknown prompt id: $want (want short or long)" >&2
      exit 2
      ;;
  esac
done

prompt_text() {
  case "$1" in
    short) printf '%s' "$PROMPT_SHORT" ;;
    long) printf '%s' "$PROMPT_LONG" ;;
  esac
}

# Portable: validate.sh dry-runs this script on macOS too, where nproc and
# sha256sum do not exist.
core_count() {
  nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown
}

sha256_of() {
  local sum=""
  if command -v sha256sum >/dev/null 2>&1; then
    sum="$(sha256sum "$1" 2>/dev/null | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    sum="$(shasum -a 256 "$1" 2>/dev/null | awk '{print $1}')"
  fi
  printf '%s' "${sum:-unknown}"
}

host_facts() {
  printf 'host: %s\n' "$(hostname)"
  printf 'kernel: %s\n' "$(uname -srm)"
  printf 'cores: %s\n' "$(core_count)"
  printf 'governor: %s\n' \
    "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  printf 'throttled: %s\n' "$(vcgencmd get_throttled 2>/dev/null || echo unknown)"
  printf 'rayon_threads: %s\n' "${NANOCAMELID_RAYON_THREADS:-${RAYON_NUM_THREADS:-unset}}"
  printf 'taskset: %s\n' "${SCARP_TASKSET:-none}"
  printf 'binary: %s\n' "$BINARY"
  printf 'binary_sha256: %s\n' "$(sha256_of "$BINARY")"
  printf 'extra_env: %s\n' "${EXTRA_ENV:-none}"
  printf 'label: %s\n' "${LABEL:-none}"
  printf 'tokens: %s\n' "$TOKENS"
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "SCARP guard dry run"
  host_facts
  echo "out_dir: $OUT_DIR"
  echo "baseline_dir: ${BASELINE_DIR:-none}"
  echo "repeats: $REPEATS"
  echo "tolerance_pct: $TOLERANCE"
  for entry in "${SELECTED_ROWS[@]}"; do
    IFS='|' read -r id gguf kind <<<"$entry"
    for p in "${SELECTED_PROMPTS[@]}"; do
      echo "run: $id/$p model=$WORKSPACE/models/$gguf head_kind=$kind"
    done
  done
  exit 0
fi

if [[ ! -x "$BINARY" ]]; then
  echo "NanoCamelid release binary not found or not executable: $BINARY" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"
host_facts >"$OUT_DIR/host.txt"

# Run one generation, sampling /proc/<pid>/status into a time series. Two
# numbers matter for SCARP and they are not the same number:
#
#   VmHWM  - process high-water mark. On this loader it peaks during weight
#            load, when the mmap pages and the decoded Vec copies are both
#            resident, so it overstates what decode actually needs.
#   decode - median VmRSS over the generation window, i.e. what is resident
#            while tokens are being produced. This is the number comparable to
#            docs/MODEL_CATALOG.md's RSS column.
#
# The series is kept per run (<log>.rss: epoch_s rss_kb hwm_kb) so the window
# can be re-derived later; do NOT use the last sample, which lands during
# teardown after the weights are dropped.
RSS_PEAK_KB=""
RSS_SERIES=""
RUN_END_EPOCH=""
run_one() {
  local log="$1"
  shift
  local pid state hwm rss peak=0
  local series="$log.rss"

  : >"$series"

  # shellcheck disable=SC2086 # EXTRA_ENV is intentionally word-split
  env NANOCAMELID_TRACE=1 $EXTRA_ENV ${SCARP_TASKSET:+taskset -c "$SCARP_TASKSET"} \
    "$BINARY" "$@" >"$log" 2>&1 &
  pid=$!

  while :; do
    state="$(awk '/^State:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    if [[ -z "$state" || "$state" == "Z" ]]; then
      break
    fi
    read -r hwm rss < <(awk '/^VmHWM:/ {h = $2} /^VmRSS:/ {r = $2} END {print h, r}' \
      "/proc/$pid/status" 2>/dev/null || true)
    if [[ "$hwm" =~ ^[0-9]+$ ]]; then
      if ((hwm > peak)); then peak="$hwm"; fi
      printf '%s %s %s\n' "${EPOCHREALTIME/,/.}" "${rss:-0}" "$hwm" >>"$series"
    fi
    sleep 0.2
  done

  set +e
  wait "$pid"
  local status=$?
  set -e
  RUN_END_EPOCH="${EPOCHREALTIME/,/.}"
  RSS_PEAK_KB="$peak"
  RSS_SERIES="$series"
  return "$status"
}

# Median VmRSS over the generation window: the last `gen_sec` seconds of the
# run, trimmed by 0.4s at the end so teardown samples are excluded.
decode_rss_kb() {
  local series="$1" gen_sec="$2" end="$3"
  [[ -f "$series" && "$gen_sec" =~ ^[0-9.]+$ ]] || return 0
  awk -v g="$gen_sec" -v e="$end" '
    ($1 >= e - g) && ($1 <= e - 0.4) { v[n++] = $2 }
    END {
      if (n == 0) exit
      for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
          if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }
      print v[int(n / 2)]
    }' "$series"
}

# The generated text is everything between the "Prompt ingested" line and the
# "Generated N tokens" line. Hash it so a later phase can prove byte-identity.
completion_sha256() {
  awk '
    /^Prompt ingested in /       { capture = 1; next }
    /^Generated [0-9]+ tokens in / { capture = 0 }
    capture                      { print }
  ' "$1" | sha256sum | awk '{print $1}'
}

trace_ms() {
  # $1 log, $2 stage -> total ms (empty when the stage did not fire)
  # Line shape: "  <stage>  calls <n> total <ms> ms avg <ms> ms"
  awk -v want="$2" '$1 == want { print $5 }' "$1" | tail -n 1
}

trace_calls() {
  awk -v want="$2" '$1 == want { print $3 }' "$1" | tail -n 1
}

json_number_or_null() {
  if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then printf '%s' "$1"; else printf 'null'; fi
}

SUMMARY="$OUT_DIR/summary.tsv"
printf 'row\tprompt\trep\tstatus\ttokens_per_sec\tgeneration_sec\tprefill_sec\tprompt_tokens\tpeak_rss_kb\tdecode_rss_kb\tforward_total_ms\tlayer_total_ms\thead_ms\thead_total_ms\thead_quant_calls\thead_pct\tcompletion_sha256\n' >"$SUMMARY"

EXIT_STATUS=0

for rep in $(seq 1 "$REPEATS"); do
  for p in "${SELECTED_PROMPTS[@]}"; do
    # Interleave rows within a repeat rather than running each row to
    # completion, so thermal drift is spread across rows instead of loading
    # onto whichever row runs last.
    for entry in "${SELECTED_ROWS[@]}"; do
      IFS='|' read -r id gguf kind <<<"$entry"
      model="$WORKSPACE/models/$gguf"
      log="$OUT_DIR/$id.$p.r$rep.log"

      if [[ ! -f "$model" ]]; then
        echo "SKIP $id/$p rep$rep: model not found: $model" >&2
        printf '%s\t%s\t%s\tmissing_model\t\t\t\t\t\t\t\t\t\t\t\t\n' "$id" "$p" "$rep" >>"$SUMMARY"
        EXIT_STATUS=4
        continue
      fi

      echo "==> $id/$p rep$rep ($kind head) $model"
      set +e
      run_one "$log" generate "$model" "$(prompt_text "$p")" 0.0 "$TOKENS"
      run_status=$?
      set -e

      if [[ "$run_status" -ne 0 ]]; then
        echo "FAIL $id/$p rep$rep: exit $run_status (see $log)" >&2
        printf '%s\t%s\t%s\tfailed\t\t\t\t\t%s\t\t\t\t\t\t\t\t\n' "$id" "$p" "$rep" "$RSS_PEAK_KB" >>"$SUMMARY"
        EXIT_STATUS="$run_status"
        continue
      fi

      tps="$(sed -nE 's/^Generated [0-9]+ tokens in [0-9.]+s \(([0-9.]+) tokens\/sec\)$/\1/p' "$log" | tail -n 1)"
      gen_sec="$(sed -nE 's/^.*"generation_sec":([0-9.]+).*$/\1/p' "$log" | tail -n 1)"
      pre_sec="$(sed -nE 's/^.*"prefill_sec":([0-9.]+).*$/\1/p' "$log" | tail -n 1)"
      ptok="$(sed -nE 's/^.*"prompt_tokens":([0-9]+).*$/\1/p' "$log" | tail -n 1)"
      fwd="$(trace_ms "$log" decode.forward_total)"
      lay="$(trace_ms "$log" decode.layer_total)"
      head_ms="$(trace_ms "$log" decode.head)"
      head_tot="$(trace_ms "$log" decode.head_total)"
      hq_calls="$(trace_calls "$log" decode.head_quant)"
      [[ -n "$hq_calls" ]] || hq_calls=0
      head_pct=""
      if [[ "$head_tot" =~ ^[0-9.]+$ && "$fwd" =~ ^[0-9.]+$ ]]; then
        head_pct="$(awk "BEGIN { if ($fwd > 0) printf \"%.2f\", 100 * $head_tot / $fwd }")"
      fi
      sha="$(completion_sha256 "$log")"
      drss="$(decode_rss_kb "$RSS_SERIES" "$gen_sec" "$RUN_END_EPOCH")"

      printf '%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$id" "$p" "$rep" "$tps" "$gen_sec" "$pre_sec" "$ptok" "$RSS_PEAK_KB" "$drss" \
        "$fwd" "$lay" "$head_ms" "$head_tot" "$hq_calls" "$head_pct" "$sha" >>"$SUMMARY"

      printf '    tok/s %s  head %s%% of decode  RSS peak %s MiB / decode %s MiB\n' \
        "${tps:-?}" "${head_pct:-?}" \
        "$(awk "BEGIN { printf \"%.0f\", ${RSS_PEAK_KB:-0} / 1024 }")" \
        "$(awk "BEGIN { printf \"%.0f\", ${drss:-0} / 1024 }")"
    done
  done
done

echo
echo "Summary written to $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

if [[ -n "$BASELINE_DIR" ]]; then
  BASE_SUMMARY="$BASELINE_DIR/summary.tsv"
  if [[ ! -f "$BASE_SUMMARY" ]]; then
    echo "Baseline summary not found: $BASE_SUMMARY" >&2
    exit 2
  fi

  echo
  echo "==> Diff vs baseline $BASELINE_DIR (tolerance ${TOLERANCE}%)"
  regressions=0
  drifts=0

  while IFS=$'\t' read -r row prompt rep status tps _gen _pre _ptok _rss _lastrss _fwd _lay _head _headtot _hq _pct sha; do
    [[ "$row" == "row" ]] && continue
    [[ "$status" == "ok" ]] || continue
    base_line="$(awk -F'\t' -v r="$row" -v p="$prompt" -v k="$rep" \
      '$1 == r && $2 == p && $3 == k && $4 == "ok" { print; exit }' "$BASE_SUMMARY")"
    if [[ -z "$base_line" ]]; then
      echo "  NEW   $row/$prompt rep$rep: no baseline entry"
      continue
    fi
    base_tps="$(awk -F'\t' '{print $5}' <<<"$base_line")"
    base_sha="$(awk -F'\t' '{print $17}' <<<"$base_line")"

    if [[ "$sha" != "$base_sha" ]]; then
      if [[ "$ALLOW_TEXT_DRIFT" == "1" ]]; then
        echo "  DRIFT $row/$prompt rep$rep: token stream changed (allowed by --allow-text-drift)"
        drifts=$((drifts + 1))
      else
        echo "  FAIL  $row/$prompt rep$rep: token stream changed ($base_sha -> $sha)"
        regressions=$((regressions + 1))
      fi
    fi

    if [[ "$tps" =~ ^[0-9.]+$ && "$base_tps" =~ ^[0-9.]+$ ]]; then
      delta="$(awk "BEGIN { printf \"%.2f\", 100 * ($tps - $base_tps) / $base_tps }")"
      if awk "BEGIN { exit !($delta < -$TOLERANCE) }"; then
        echo "  FAIL  $row/$prompt rep$rep: tok/s $base_tps -> $tps (${delta}%)"
        regressions=$((regressions + 1))
      else
        echo "  ok    $row/$prompt rep$rep: tok/s $base_tps -> $tps (${delta}%)"
      fi
    fi
  done <"$SUMMARY"

  echo "json: {\"guard\":\"scarp\",\"baseline\":$(printf '"%s"' "$BASELINE_DIR"),\"regressions\":$regressions,\"allowed_drifts\":$drifts}"
  if ((regressions > 0)); then
    echo "scarp_guard_status: regressed"
    exit 5
  fi
fi

if [[ "$EXIT_STATUS" -eq 0 ]]; then
  echo "scarp_guard_status: ok"
fi
exit "$EXIT_STATUS"
