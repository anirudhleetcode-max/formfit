#!/usr/bin/env bash
# Starts backend (:8003) + built frontend (vite preview :5175), runs the Playwright test, stops both.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PYTHON:-python3}"
mkdir -p "$HERE/.cache"
LOG="$HERE/.cache"

# fake webcam: Chromium plays a raw .y4m file in a loop
Y4M="$HERE/.cache/squat.y4m"
if [ ! -s "$Y4M" ]; then
  ffmpeg -v error -y -i "$ROOT/samples/squat_demo.webm" -vf "scale=640:360,fps=30" -pix_fmt yuv420p "$Y4M"
fi

# refuse to run against servers that are already up (they may be stale builds)
for url in http://127.0.0.1:8003/api/health http://127.0.0.1:5175/; do
  if curl -s -o /dev/null "$url"; then echo "port for $url is already in use; stop that server first"; exit 1; fi
done

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

(cd "$ROOT/backend" && exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8003 >"$LOG/backend.log" 2>&1) &
PIDS+=($!)
(cd "$ROOT/frontend" && npm run build >"$LOG/build.log" 2>&1)
(cd "$ROOT/frontend" && exec node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 5175 --strictPort >"$LOG/frontend.log" 2>&1) &
PIDS+=($!)

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8003/api/health >/dev/null && curl -sf http://127.0.0.1:5175/ >/dev/null; then break; fi
  sleep 1
done
curl -sf http://127.0.0.1:8003/api/health || { echo "backend did not start"; cat "$LOG/backend.log"; exit 1; }

(cd "$ROOT/backend" && "$PY" -m scripts.seed)
status=0
FAKE_CAM="$Y4M" "$PY" -m pytest "$HERE/test_e2e.py" -v -s "$@" || status=$?

# remove the throwaway e2e accounts (and their sessions) created by the test
(cd "$ROOT/backend" && "$PY" - <<'PYEOF'
import pymongo
from app.config import get_settings
s = get_settings()
db = pymongo.MongoClient(s.mongo_uri)[s.mongo_db]
ids = [u["_id"] for u in db.users.find({"email": {"$regex": "^e2e_.*@example\\.com$"}}, {"_id": 1})]
db.sessions.delete_many({"user_id": {"$in": ids}})
db.users.delete_many({"_id": {"$in": ids}})
print(f"cleaned up {len(ids)} e2e users")
PYEOF
)
exit $status
