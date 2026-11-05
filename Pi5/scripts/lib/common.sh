# HeteroServe -- shared shell helpers for the Pi 5 node scripts.
# Sourced, never executed:  . "$(dirname "$0")/lib/common.sh"
set -euo pipefail

HS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-$0}")" && pwd)"
HS_NODE_DIR="$(cd "${HS_SCRIPT_DIR}/.." && pwd)"
HS_CONFIG="${HS_NODE_DIR}/config/node.env"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
log()  { printf '%s[%s]%s %s\n' "$c_dim" "$(date -u +%H:%M:%SZ)" "$c_off" "$*"; }
ok()   { printf '%s  ok %s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s warn%s %s\n' "$c_ylw" "$c_off" "$*" >&2; }
die()  { printf '%sFAIL%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

load_env() {
  [ -f "$HS_CONFIG" ] || die "missing $HS_CONFIG"
  # shellcheck disable=SC1090
  set -a; . "$HS_CONFIG"; set +a
  : "${NODE_ID:?node.env did not define NODE_ID}"
}

need_root() { [ "$(id -u)" -eq 0 ] || die "run as root: sudo $0 $*"; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

# Refuse to run on anything that is not the board this node.env describes. Cheap
# insurance against pasting the Pi 5 commands into the Pi 4 shell.
assert_this_board() {
  local model
  model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
  case "$model" in
    *"Raspberry Pi 5"*) ok "board: $model" ;;
    *) die "expected a Raspberry Pi 5, /proc/device-tree/model says: $model" ;;
  esac
}

# "nan" when the zone cannot be read, never a number. The old fallback was `|| echo 0`,
# which yielded 0.0 -- a physically impossible SoC reading that is nonetheless a well-formed
# float, so assert_cool_enough below happily passed it (0.0 <= 55) and the R5 start gate
# disarmed itself for the whole sweep, silently. Pi 4 yields "nan" for the same reason.
soc_temp_c() {
  local raw
  raw="$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)" || raw=""
  case "$raw" in
    ''|*[!0-9-]*) echo "nan"; return 0 ;;
  esac
  awk -v r="$raw" 'BEGIN{printf "%.1f", r/1000}'
}

# The firmware's own record of under-voltage and thermal capping, as the raw vcgencmd word
# ("0x0", "0x50005"). "unavailable" when it cannot be read -- NOT "0x0". Pi 4's shell helper
# returns 0x0 when vcgencmd is missing, which reads as "clean" and is the fail-open its own
# audit flagged; the sentinel here makes the parser raise `throttle-unknown` instead, so a
# cell whose throttle state was never checked is excluded rather than published as fine.
throttled_word() {
  local out
  if ! command -v vcgencmd >/dev/null 2>&1; then echo "unavailable"; return 0; fi
  out="$(vcgencmd get_throttled 2>/dev/null)" || { echo "unavailable"; return 0; }
  case "$out" in
    throttled=0x*) echo "${out#throttled=}" ;;
    *) echo "unavailable" ;;
  esac
}

# Hard gate used by every measurement entry point: a cell must not start warm, or
# the thermal history contaminates the comparison (proposal risk R5).
assert_cool_enough() {
  local t max
  t="$(soc_temp_c)"; max="${THERMAL_START_MAX_C:-55}"
  # Fail closed on an unreadable sensor. Without a temperature this gate cannot be
  # evaluated at all, and a sweep that runs ungated produces rows whose thermal validity
  # nobody can establish afterwards -- which is worse than not running.
  if [ "$t" = "nan" ]; then
    die "cannot read /sys/class/thermal/thermal_zone0/temp -- the ${max} C start gate
  cannot be evaluated, so this run would be unvalidatable. Fix the thermal zone
  (check the path exists and is readable) before measuring."
  fi
  awk -v t="$t" -v m="$max" 'BEGIN{exit !(t+0 > 0 && t+0 <= m+0)}' \
    || die "SoC at ${t} C, outside the (0, ${max}] C start gate -- let it settle before measuring"
  ok "SoC at ${t} C (gate ${max} C)"
}
