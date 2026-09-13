"""
Persistent Observation Database & Ingestion Layer for Sagar-Drishti V2.3 Shadow Evaluation.

CRITICAL SPECIFICATIONS:
1. Research Only: Completely isolated from operational database and production state.
2. Deterministic Observation IDs: Unique hash prevents duplicate record inflation.
3. Append-Only History: Ingestion handles duplicates gracefully (INSERT OR IGNORE).
4. Safe Ingestion: Corrupted or malformed lines are isolated without failing the batch.
5. Thread-Safe: Uses SQLite in WAL mode with robust transaction handling.
"""

import os
import json
import sqlite3
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("sagar_drishti.shadow_v2_3_store")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHADOW_DATA_DIR = os.path.join(BASE_DIR, "data", "shadow")
DEFAULT_DB_PATH = os.path.join(SHADOW_DATA_DIR, "shadow_v2_3_observations.db")

os.makedirs(SHADOW_DATA_DIR, exist_ok=True)


def compute_observation_id(
    prediction_timestamp: str,
    site: str,
    horizon: int,
    ocean_timestamp: Optional[str] = None,
) -> str:
    """
    Computes a deterministic observation ID based on primary invariant keys.
    Ensures identical telemetry records generate identical IDs.
    """
    norm_ts = str(prediction_timestamp).strip()
    norm_site = str(site).strip().lower()
    norm_h = str(horizon).strip()
    norm_ocean = str(ocean_timestamp or "").strip()
    raw_key = f"{norm_ts}_{norm_site}_{norm_h}_{norm_ocean}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return f"obs_{digest[:16]}"


def compute_atmospheric_age_hours(
    prediction_timestamp: Optional[str],
    atmosphere_timestamp: Optional[str],
) -> Optional[float]:
    """
    Calculates atmospheric data age (in hours) at the time of prediction.
    """
    if not prediction_timestamp or not atmosphere_timestamp:
        return None
    try:
        pred_clean = prediction_timestamp.replace("Z", "+00:00")
        atmos_clean = atmosphere_timestamp.replace("Z", "+00:00")
        dt_pred = datetime.fromisoformat(pred_clean)
        dt_atmos = datetime.fromisoformat(atmos_clean)
        age_hours = (dt_pred - dt_atmos).total_seconds() / 3600.0
        return round(age_hours, 2)
    except Exception:
        return None


class ShadowV23Store:
    """
    Persistent SQLite storage layer for V2.3 shadow observations and storm events.
    Thread-safe, read/write isolated, deterministic.
    """

    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Initializes database schema and indexes."""
        with self._lock:
            with self._get_connection() as conn:
                # 1. Observations Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_observations (
                        observation_id TEXT PRIMARY KEY,
                        prediction_timestamp TEXT NOT NULL,
                        prediction_date TEXT NOT NULL,
                        site TEXT NOT NULL,
                        horizon INTEGER NOT NULL,
                        production_model_version TEXT,
                        shadow_model_version TEXT,
                        production_probability REAL,
                        shadow_probability REAL,
                        probability_delta REAL,
                        production_alert TEXT,
                        shadow_research_threshold_result TEXT,
                        ocean_timestamp TEXT,
                        atmosphere_timestamp TEXT,
                        atmospheric_data_age_hours REAL,
                        data_quality_status TEXT NOT NULL,
                        coverage_status TEXT,
                        inference_status TEXT NOT NULL,
                        failure_reason TEXT,
                        model_artifact_hash TEXT,
                        feature_schema_hash TEXT,
                        created_at TEXT NOT NULL,
                        raw_record_json TEXT
                    );
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_site_horizon ON shadow_observations (site, horizon);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_pred_date ON shadow_observations (prediction_date);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_dq_status ON shadow_observations (data_quality_status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_inf_status ON shadow_observations (inference_status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_pred_ts ON shadow_observations (prediction_timestamp);")

                # 2. Storm Events Registry Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_events (
                        event_id TEXT PRIMARY KEY,
                        event_name TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        basin TEXT NOT NULL,
                        max_intensity_kts REAL,
                        classification TEXT,
                        source TEXT,
                        is_verified INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL
                    );
                """)

                # 3. Malformed Records Quarantine Table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS shadow_malformed_telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_file TEXT,
                        line_number INTEGER,
                        raw_content TEXT,
                        error_reason TEXT,
                        captured_at TEXT NOT NULL
                    );
                """)

                conn.commit()

        # Seed canonical reference historical events if not present
        self._seed_reference_events()

    def _seed_reference_events(self):
        """
        Seeds canonical reference events from the 10-year / test split
        for historical context (read-only reference; quarantined from tuning).
        """
        ref_events = [
            {
                "event_id": "STORM-2025-MONTHA",
                "event_name": "Montha",
                "start_date": "2025-10-24",
                "end_date": "2025-10-30",
                "basin": "bob",
                "max_intensity_kts": 50.0,
                "classification": "SCS",
                "source": "IMD Historical (SD-V2.3 Test Split Reference)",
            },
            {
                "event_id": "STORM-2025-UNNAMED-12",
                "event_name": "UNNAMED-12",
                "start_date": "2025-11-28",
                "end_date": "2025-12-02",
                "basin": "bob",
                "max_intensity_kts": 25.0,
                "classification": "D",
                "source": "IMD Historical (SD-V2.3 Test Split Reference)",
            },
        ]
        with self._get_connection() as conn:
            for ev in ref_events:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_events (
                        event_id, event_name, start_date, end_date, basin,
                        max_intensity_kts, classification, source, is_verified, created_at
                    ) VALUES (
                        :event_id, :event_name, :start_date, :end_date, :basin,
                        :max_intensity_kts, :classification, :source, 1, datetime('now')
                    );
                    """,
                    ev,
                )
            conn.commit()

    def insert_observation(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Inserts a single shadow telemetry record.
        Returns (is_inserted, observation_id).
        If record already exists (duplicate), returns (False, observation_id).
        """
        if not isinstance(record, dict):
            raise ValueError("Telemetry record must be a dictionary")

        pred_ts = record.get("prediction_timestamp") or record.get("timestamp")
        if not pred_ts:
            raise ValueError("Record missing required timestamp")

        site = str(record.get("site", "unknown")).lower()
        horizon = int(record.get("horizon", 3))
        ocean_ts = record.get("ocean_timestamp")
        obs_id = compute_observation_id(pred_ts, site, horizon, ocean_ts)

        pred_date = record.get("prediction_date")
        if not pred_date:
            if ocean_ts:
                pred_date = ocean_ts[:10]
            else:
                pred_date = pred_ts[:10]

        prod_prob = record.get("production_probability")
        shadow_prob = record.get("shadow_probability")
        prob_delta = None
        if prod_prob is not None and shadow_prob is not None:
            try:
                prob_delta = round(float(shadow_prob) - float(prod_prob), 6)
            except (ValueError, TypeError):
                pass

        atmos_ts = record.get("atmosphere_timestamp")
        data_age = record.get("atmospheric_data_age_hours")
        if data_age is None and atmos_ts:
            data_age = compute_atmospheric_age_hours(pred_ts, atmos_ts)

        raw_json = json.dumps(record)
        created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_observations (
                        observation_id, prediction_timestamp, prediction_date, site, horizon,
                        production_model_version, shadow_model_version, production_probability,
                        shadow_probability, probability_delta, production_alert,
                        shadow_research_threshold_result, ocean_timestamp, atmosphere_timestamp,
                        atmospheric_data_age_hours, data_quality_status, coverage_status,
                        inference_status, failure_reason, model_artifact_hash, feature_schema_hash,
                        created_at, raw_record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        obs_id,
                        pred_ts,
                        pred_date,
                        site,
                        horizon,
                        record.get("production_model_version", "v1.1.0"),
                        record.get("shadow_model_version", "v2.3.0-model-d-shadow"),
                        float(prod_prob) if prod_prob is not None else None,
                        float(shadow_prob) if shadow_prob is not None else None,
                        prob_delta,
                        record.get("production_alert", "NO_ALERT"),
                        record.get("shadow_alert_at_research_threshold") or record.get("shadow_research_threshold_result"),
                        ocean_ts,
                        atmos_ts,
                        data_age,
                        record.get("data_quality_status", "UNKNOWN"),
                        record.get("coverage_status"),
                        record.get("inference_status", "UNKNOWN"),
                        record.get("failure_reason"),
                        record.get("model_artifact_hash"),
                        record.get("feature_schema_hash"),
                        created_at,
                        raw_json,
                    ),
                )
                conn.commit()
                inserted = cursor.rowcount > 0
                return inserted, obs_id

    def ingest_jsonl_file(self, file_path: str) -> Dict[str, Any]:
        """
        Safely ingests a JSONL telemetry file into SQLite.
        Handles duplicates, isolates malformed rows without crashing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Telemetry file not found: {file_path}")

        total_lines = 0
        inserted_count = 0
        duplicate_count = 0
        malformed_count = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw_line = line.strip()
                if not raw_line:
                    continue
                total_lines += 1
                try:
                    record = json.loads(raw_line)
                    if not isinstance(record, dict):
                        raise ValueError("Parsed JSON is not an object")
                    inserted, _ = self.insert_observation(record)
                    if inserted:
                        inserted_count += 1
                    else:
                        duplicate_count += 1
                except Exception as e:
                    malformed_count += 1
                    self._record_malformed(file_path, line_no, raw_line, str(e))

        return {
            "file": file_path,
            "total_lines": total_lines,
            "inserted": inserted_count,
            "duplicates": duplicate_count,
            "malformed": malformed_count,
        }

    def ingest_all_shadow_telemetry(self, shadow_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Discovers and ingests all shadow_v2_3_telemetry_*.jsonl files in data/shadow.
        """
        target_dir = shadow_dir or SHADOW_DATA_DIR
        if not os.path.exists(target_dir):
            return {"total_files": 0, "total_inserted": 0, "total_duplicates": 0, "total_malformed": 0}

        results = []
        tot_ins = 0
        tot_dup = 0
        tot_mal = 0

        for fname in sorted(os.listdir(target_dir)):
            if fname.startswith("shadow_v2_3_telemetry_") and fname.endswith(".jsonl"):
                full_path = os.path.join(target_dir, fname)
                res = self.ingest_jsonl_file(full_path)
                results.append(res)
                tot_ins += res["inserted"]
                tot_dup += res["duplicates"]
                tot_mal += res["malformed"]

        return {
            "total_files": len(results),
            "files": results,
            "total_inserted": tot_ins,
            "total_duplicates": tot_dup,
            "total_malformed": tot_mal,
        }

    def _record_malformed(self, file_path: str, line_number: int, content: str, reason: str):
        """Quarantines malformed telemetry line for audit."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO shadow_malformed_telemetry (source_file, line_number, raw_content, error_reason, captured_at)
                    VALUES (?, ?, ?, ?, datetime('now'));
                    """,
                    (file_path, line_number, content[:2000], reason),
                )
                conn.commit()
        except Exception as e:
            logger.debug("Failed to quarantine malformed telemetry: %s", e)

    def register_event(
        self,
        event_id: str,
        event_name: str,
        start_date: str,
        end_date: str,
        basin: str,
        max_intensity_kts: Optional[float] = None,
        classification: Optional[str] = None,
        source: Optional[str] = None,
        is_verified: int = 1,
    ) -> bool:
        """
        Registers an independent storm event in the event registry.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR REPLACE INTO shadow_events (
                        event_id, event_name, start_date, end_date, basin,
                        max_intensity_kts, classification, source, is_verified, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'));
                    """,
                    (
                        event_id,
                        event_name,
                        start_date,
                        end_date,
                        basin.lower(),
                        max_intensity_kts,
                        classification,
                        source or "Independent Observation",
                        is_verified,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0

    def query_observations(
        self,
        site: Optional[str] = None,
        horizon: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_quality_status: Optional[str] = None,
        inference_status: Optional[str] = None,
        limit: int = 5000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Queries shadow observations with flexible filtering.
        """
        conditions = []
        params: List[Any] = []

        if site:
            conditions.append("site = ?")
            params.append(site.lower())
        if horizon is not None:
            conditions.append("horizon = ?")
            params.append(int(horizon))
        if start_date:
            conditions.append("prediction_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("prediction_date <= ?")
            params.append(end_date)
        if data_quality_status:
            conditions.append("data_quality_status = ?")
            params.append(data_quality_status)
        if inference_status:
            conditions.append("inference_status = ?")
            params.append(inference_status)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM shadow_observations
            {where_clause}
            ORDER BY prediction_timestamp ASC
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_observation_counts(self) -> Dict[str, Any]:
        """Returns overall database counts and health summary."""
        with self._get_connection() as conn:
            tot = conn.execute("SELECT COUNT(*) FROM shadow_observations;").fetchone()[0]
            succ = conn.execute("SELECT COUNT(*) FROM shadow_observations WHERE inference_status = 'SUCCESS';").fetchone()[0]
            fail = conn.execute("SELECT COUNT(*) FROM shadow_observations WHERE inference_status = 'FAILED';").fetchone()[0]
            skip = conn.execute("SELECT COUNT(*) FROM shadow_observations WHERE inference_status = 'SKIPPED' OR inference_status = 'DROPPED';").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM shadow_events;").fetchone()[0]
            malformed = conn.execute("SELECT COUNT(*) FROM shadow_malformed_telemetry;").fetchone()[0]

            return {
                "total_observations": tot,
                "successful_inferences": succ,
                "failed_inferences": fail,
                "skipped_or_dropped": skip,
                "registered_events": events,
                "malformed_telemetry_lines": malformed,
            }

    def get_events(self) -> List[Dict[str, Any]]:
        """Returns all registered storm events."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM shadow_events ORDER BY start_date ASC;")
            return [dict(row) for row in cursor.fetchall()]
