# uso: source scripts/load_env.sh && load_env "$PY"
load_env() {
  local py="${1:-python}" env_file="${2:-.env}" out
  [ -f "$env_file" ] || return 0
  out="$("$py" - "$env_file" <<'PYEOF'
import shlex, sys
from dotenv import dotenv_values
for k, v in dotenv_values(sys.argv[1]).items():
    if v is not None:
        print(f"export {k}={shlex.quote(v)}")
PYEOF
)" || { echo "erro ao ler $env_file (python-dotenv instalado em $py?)" >&2; return 1; }
  eval "$out"
}
