import time
from datetime import datetime, timezone
from typing import Optional

SEVERITY_POINTS = {
    "LOW":    5,
    "MEDIUM": 15,
    "HIGH":   30,
}

DECAY_RATE = 2  # points per second of clean behavior

STATUS_THRESHOLDS = {
    "CLEAN":     (0,  30),
    "SUSPICIOUS":(31, 60),
    "HIGH_RISK": (61, 90),
    "CRITICAL":  (91, 100),
}


class AlertEngine:
    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        self.risk_score: float = 0.0
        self.score_history: list[dict] = []   # [{time, score}]
        self.last_alert: Optional[dict] = None
        self._last_alert_type: Optional[str] = None
        self._last_alert_time: float = 0.0

    # ------------------------------------------------------------------
    def add_alert(
        self,
        alert_type: str,
        severity: str,
        extra: dict | None = None,
    ) -> dict:
        now = time.time()

        # Debounce: same alert type no more than once per 2s
        if (
            alert_type == self._last_alert_type
            and (now - self._last_alert_time) < 2.0
        ):
            # Still update score softly
            self.risk_score = min(100, self.risk_score + SEVERITY_POINTS.get(severity, 5) * 0.2)
            return None  # suppressed

        points = SEVERITY_POINTS.get(severity, 5)
        self.risk_score = min(100.0, self.risk_score + points)

        ts = datetime.now(timezone.utc).isoformat()

        alert = {
            "candidate_id": self.candidate_id,
            "type":         alert_type,
            "severity":     severity,
            "score":        int(self.risk_score),
            "timestamp":    ts,
            "snapshot_url": None,
        }
        if extra:
            alert.update(extra)

        self.last_alert = alert
        self._last_alert_type = alert_type
        self._last_alert_time = now

        self.score_history.append({
            "time":  ts,
            "score": int(self.risk_score),
        })

        # Keep only last 200 entries
        if len(self.score_history) > 200:
            self.score_history = self.score_history[-200:]

        return alert

    # ------------------------------------------------------------------
    def decay(self, elapsed_seconds: float):
        """Call this every frame with time since last frame."""
        self.risk_score = max(0.0, self.risk_score - DECAY_RATE * elapsed_seconds)

    # ------------------------------------------------------------------
    def get_status(self) -> str:
        score = int(self.risk_score)
        if score > 90:
            return "CRITICAL"
        elif score > 60:
            return "HIGH_RISK"
        elif score > 30:
            return "SUSPICIOUS"
        else:
            return "CLEAN"
