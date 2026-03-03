# ── Helpers ───────────────────────────────────────────────────────────────────
import uuid
from datetime import datetime
def _make_request_id() -> str:
    return str(uuid.uuid4())

def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"

