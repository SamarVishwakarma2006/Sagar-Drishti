"""
Canonical 101-Feature Schema Manifest & Integrity Engine for Sagar-Drishti.
Defines the frozen 101-feature contract, variable sources, transformations,
rolling temporal windows, physical units, and valid operating ranges.
"""

import json
from typing import Dict, Any, List

SUPPORTED_CHANNELS = ["temp", "sal", "cur_u", "cur_v", "cur", "ssh", "mld"]

CHANNEL_METADATA: Dict[str, Dict[str, Any]] = {
    "temp": {
        "source_variable": "thetao",
        "long_name": "Sea Water Potential Temperature",
        "unit": "°C",
        "has_pct_anom": False,
        "valid_range": [10.0, 38.0],
    },
    "sal": {
        "source_variable": "so",
        "long_name": "Sea Water Practical Salinity",
        "unit": "PSU",
        "has_pct_anom": True,
        "valid_range": [0.0, 45.0],
    },
    "cur_u": {
        "source_variable": "uo",
        "long_name": "Eastward Sea Water Velocity",
        "unit": "m/s",
        "has_pct_anom": False,
        "valid_range": [-4.0, 4.0],
    },
    "cur_v": {
        "source_variable": "vo",
        "long_name": "Northward Sea Water Velocity",
        "unit": "m/s",
        "has_pct_anom": False,
        "valid_range": [-4.0, 4.0],
    },
    "cur": {
        "source_variable": "sqrt(uo^2 + vo^2)",
        "long_name": "Ocean Surface Current Speed",
        "unit": "m/s",
        "has_pct_anom": True,
        "valid_range": [0.0, 5.0],
    },
    "ssh": {
        "source_variable": "zos",
        "long_name": "Sea Surface Height Above Geoid",
        "unit": "m",
        "has_pct_anom": False,
        "valid_range": [-3.0, 3.0],
    },
    "mld": {
        "source_variable": "mlotst",
        "long_name": "Ocean Mixed Layer Thickness",
        "unit": "m",
        "has_pct_anom": True,
        "valid_range": [1.0, 300.0],
    },
}

ROLLING_WINDOWS = [("7d", 7), ("14d", 14), ("30d", 30)]


class FeatureManifest:
    """
    Generates and audits the hard contract for Sagar-Drishti's 101 features.
    """

    @classmethod
    def get_canonical_101_feature_names(cls) -> List[str]:
        """Returns deterministic list of exactly 101 feature names in hard contract order."""
        features = []
        for ch in SUPPORTED_CHANNELS:
            meta = CHANNEL_METADATA[ch]
            # 1. Current value
            features.append(f"{ch}_current")
            # 2. Climatological baseline statistics
            features.append(f"{ch}_base_mean")
            features.append(f"{ch}_base_std")
            # 3. Baseline anomalies
            features.append(f"{ch}_abs_anom")
            features.append(f"{ch}_zscore")
            if meta["has_pct_anom"]:
                features.append(f"{ch}_pct_anom")
            # 4. Rolling window dynamics
            for w_key, _ in ROLLING_WINDOWS:
                features.append(f"{ch}_{w_key}_mean")
                features.append(f"{ch}_{w_key}_delta")
                features.append(f"{ch}_{w_key}_trend")
        return features

    @classmethod
    def generate_manifest(cls) -> Dict[str, Any]:
        """Produces full JSON-serializable schema manifest for all 101 features."""
        names = cls.get_canonical_101_feature_names()
        manifest_entries = []

        for name in names:
            # Parse variable prefix
            ch = next(c for c in SUPPORTED_CHANNELS if name.startswith(f"{c}_"))
            meta = CHANNEL_METADATA[ch]
            suffix = name[len(ch) + 1:]

            if suffix == "current":
                trans = "Direct surface measurement / observation at date T"
                win = "0d (current)"
                unit = meta["unit"]
                v_range = meta["valid_range"]
            elif suffix == "base_mean":
                trans = "Climatological mean across historical period"
                win = "full_period"
                unit = meta["unit"]
                v_range = meta["valid_range"]
            elif suffix == "base_std":
                trans = "Climatological standard deviation across historical period"
                win = "full_period"
                unit = meta["unit"]
                v_range = [0.0, max(meta["valid_range"])]
            elif suffix == "abs_anom":
                trans = "Absolute anomaly: value(T) - base_mean"
                win = "instantaneous vs baseline"
                unit = meta["unit"]
                v_range = [-max(meta["valid_range"]), max(meta["valid_range"])]
            elif suffix == "zscore":
                trans = "Standardized anomaly: (value(T) - base_mean) / base_std"
                win = "instantaneous vs baseline"
                unit = "z-score (dimensionless)"
                v_range = [-10.0, 10.0]
            elif suffix == "pct_anom":
                trans = "Percentage anomaly: (abs_anom / base_mean) * 100"
                win = "instantaneous vs baseline"
                unit = "%"
                v_range = [-500.0, 500.0]
            elif "_mean" in suffix:
                w_key = suffix.split("_")[0]
                trans = f"Rolling mean over preceding {w_key} window"
                win = w_key
                unit = meta["unit"]
                v_range = meta["valid_range"]
            elif "_delta" in suffix:
                w_key = suffix.split("_")[0]
                trans = f"Window net change: value(T) - value(T - {w_key})"
                win = w_key
                unit = meta["unit"]
                v_range = [-max(meta["valid_range"]), max(meta["valid_range"])]
            elif "_trend" in suffix:
                w_key = suffix.split("_")[0]
                trans = f"Linear slope / rate of change per day over preceding {w_key}"
                win = w_key
                unit = f"{meta['unit']}/day"
                v_range = [-5.0, 5.0]
            else:
                trans = "Physical transformation"
                win = "custom"
                unit = meta["unit"]
                v_range = None

            manifest_entries.append({
                "feature_name": name,
                "ocean_channel": ch,
                "source_copernicus_variable": meta["source_variable"],
                "variable_long_name": meta["long_name"],
                "transformation": trans,
                "temporal_window": win,
                "unit": unit,
                "valid_range": v_range,
                "is_numeric": True,
            })

        return {
            "schema_name": "Sagar-Drishti Canonical 101-Feature Ocean Contract",
            "version": "1.1.0",
            "total_features": len(manifest_entries),
            "channels_count": len(SUPPORTED_CHANNELS),
            "channels": SUPPORTED_CHANNELS,
            "feature_manifest": manifest_entries,
        }

    @classmethod
    def audit_feature_vector(cls, features: List[str]) -> Tuple[bool, List[str]]:
        """Verifies an incoming feature vector against the hard 101 contract."""
        canonical = cls.get_canonical_101_feature_names()
        errors = []
        if len(features) != 101:
            errors.append(f"Expected exactly 101 features, got {len(features)}")
        if features != canonical:
            missing = [f for f in canonical if f not in features]
            extra = [f for f in features if f not in canonical]
            if missing:
                errors.append(f"Missing canonical features: {missing[:5]}...")
            if extra:
                errors.append(f"Unexpected extra features: {extra[:5]}...")
            if not missing and not extra and features != canonical:
                errors.append("Feature ordering deviates from the canonical hard contract.")
        return len(errors) == 0, errors

    @classmethod
    def validate_feature_list(cls, features: List[str]) -> None:
        """Raises ValueError if features do not strictly match the 101-feature contract."""
        ok, errors = cls.audit_feature_vector(features)
        if not ok:
            raise ValueError(f"Feature list violates canonical 101 contract: {errors}")

