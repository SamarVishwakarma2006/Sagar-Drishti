import io
import re
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ..models.schemas import BoundingBox, VariableStats, SitePhysics, FloatRecord, ProfileResult, ProfilePoint


class TabularParser:
    """
    Parses Argo float, Glider, CTD, and in-situ oceanographic data from .csv and .txt files.
    Auto-detects headers (Coriolis, INCOIS, Ifremer, WOCE, standard CSV), extracts vertical
    depth profiles, and generates OGC-compliant GeoJSON structures.
    """

    LAT_PATTERNS = [r"^lat", r"^latitude", r"^nav_lat", r"^y$"]
    LON_PATTERNS = [r"^lon", r"^long", r"^longitude", r"^nav_lon", r"^x$"]
    DEPTH_PATTERNS = [r"^depth", r"^pres", r"^pressure", r"^ctdprs", r"^level", r"^z$"]
    TEMP_PATTERNS = [r"^temp", r"^temperature", r"^ctdtmp", r"^theta", r"^sst", r"^thetao$"]
    SAL_PATTERNS = [r"^sal", r"^salinity", r"^psal", r"^ctdsal", r"^sss", r"^salt", r"^so$"]
    OXY_PATTERNS = [r"^oxy", r"^oxygen", r"^doxy", r"^ctdoxy", r"^dox2", r"^o2"]
    CUR_PATTERNS = [r"^cur", r"^current", r"^speed", r"^cur_speed", r"^velocity"]
    DIR_PATTERNS = [r"^dir", r"^direction", r"^cur_dir", r"^heading"]
    # Copernicus-specific velocity components
    CUR_U_PATTERNS = [r"^uo$", r"^cur_u", r"^u_vel", r"^eastward"]
    CUR_V_PATTERNS = [r"^vo$", r"^cur_v", r"^v_vel", r"^northward"]
    # Copernicus SSH and MLD
    SSH_PATTERNS = [r"^zos$", r"^ssh", r"^sea_surface_height", r"^sla"]
    MLD_PATTERNS = [r"^mlotst$", r"^mld", r"^mixed_layer", r"^mld_"]
    ID_PATTERNS = [r"^platform", r"^float", r"^id", r"^wmo", r"^station", r"^expocode", r"^float_id"]
    CYCLE_PATTERNS = [r"^cycle", r"^cast", r"^profile", r"^station_number"]
    TIME_PATTERNS = [r"^time", r"^date", r"^datetime", r"^juld", r"^timestamp"]

    @classmethod
    def _match_column(cls, col_name: str, patterns: List[str]) -> bool:
        clean = col_name.strip().lower().replace(" ", "_").replace("-", "_")
        for p in patterns:
            if re.search(p, clean):
                return True
        return False

    @classmethod
    def _find_column(cls, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        for col in df.columns:
            if cls._match_column(str(col), patterns):
                return str(col)
        return None

    @classmethod
    def parse_csv_from_bytes(cls, content: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parses raw CSV/TXT bytes, auto-detecting delimiters and header row.
        """
        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()

        # Detect and skip comments (lines starting with #, //, %, *, @)
        skip_count = 0
        for line in lines[:30]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "%", "*", "@")):
                skip_count += 1
            else:
                break

        clean_text = "\n".join(lines[skip_count:])

        # Detect delimiter (comma, semicolon, tab, or whitespace)
        first_line = lines[skip_count] if len(lines) > skip_count else ""
        if "\t" in first_line:
            sep = "\t"
        elif ";" in first_line:
            sep = ";"
        elif "," in first_line:
            sep = ","
        else:
            sep = r"\s+"

        try:
            df = pd.read_csv(io.StringIO(clean_text), sep=sep, engine="python")
        except Exception as e:
            # Fallback to standard comma read_csv
            df = pd.read_csv(io.StringIO(clean_text), on_bad_lines="skip")

        if df.empty or len(df.columns) < 2:
            raise ValueError(f"Tabular dataset '{filename}' has invalid shape or empty data.")

        metadata = cls.extract_metadata(df, filename)
        return df, metadata

    @classmethod
    def extract_metadata(cls, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Extracts Argo float profiles, bounding boxes, physical variables, and creates SitePhysics.
        """
        lat_col = cls._find_column(df, cls.LAT_PATTERNS)
        lon_col = cls._find_column(df, cls.LON_PATTERNS)
        depth_col = cls._find_column(df, cls.DEPTH_PATTERNS)
        temp_col = cls._find_column(df, cls.TEMP_PATTERNS)
        sal_col = cls._find_column(df, cls.SAL_PATTERNS)
        oxy_col = cls._find_column(df, cls.OXY_PATTERNS)
        cur_col = cls._find_column(df, cls.CUR_PATTERNS)
        dir_col = cls._find_column(df, cls.DIR_PATTERNS)
        id_col = cls._find_column(df, cls.ID_PATTERNS)
        cycle_col = cls._find_column(df, cls.CYCLE_PATTERNS)
        time_col = cls._find_column(df, cls.TIME_PATTERNS)
        # Copernicus-specific component columns
        cur_u_col = cls._find_column(df, cls.CUR_U_PATTERNS)
        cur_v_col = cls._find_column(df, cls.CUR_V_PATTERNS)
        ssh_col = cls._find_column(df, cls.SSH_PATTERNS)
        mld_col = cls._find_column(df, cls.MLD_PATTERNS)

        if not lat_col or not lon_col:
            raise ValueError(f"Could not identify Latitude/Longitude columns in tabular file '{filename}'. Found headers: {list(df.columns)}")

        # Convert numeric coordinates
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])

        if depth_col:
            df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0).abs()
        else:
            depth_col = "_depth"
            df[depth_col] = 0.0

        for col in [temp_col, sal_col, oxy_col, cur_col, dir_col]:
            if col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        min_lat = float(df[lat_col].min())
        max_lat = float(df[lat_col].max())
        min_lon = float(df[lon_col].min())
        max_lon = float(df[lon_col].max())
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        max_depth = float(df[depth_col].max()) if depth_col in df else 2000.0

        # Group into float profiles
        floats: List[FloatRecord] = []
        group_cols = []
        if id_col:
            group_cols.append(id_col)
        if cycle_col:
            group_cols.append(cycle_col)

        if not group_cols:
            # Group by unique rounded lat/lon if no ID column exists
            df["_group_id"] = df[lat_col].round(3).astype(str) + "_" + df[lon_col].round(3).astype(str)
            group_cols = ["_group_id"]

        groups = df.groupby(group_cols)
        float_idx = 1

        for g_key, group_df in groups:
            g_sorted = group_df.sort_values(by=depth_col)
            flat = float(g_sorted[lat_col].iloc[0])
            flon = float(g_sorted[lon_col].iloc[0])

            plat_id = str(g_key[0] if isinstance(g_key, tuple) else g_key)
            if plat_id.startswith("290") or plat_id.isdigit():
                clean_id = f"Argo_{plat_id}"
            else:
                clean_id = f"Float_{float_idx}_{plat_id[:8]}"

            cycle_val = int(g_sorted[cycle_col].iloc[0]) if cycle_col and not np.isnan(g_sorted[cycle_col].iloc[0]) else float_idx

            profile_points: List[ProfilePoint] = []
            for _, row in g_sorted.iterrows():
                z = float(row[depth_col])
                t_val = float(row[temp_col]) if temp_col and not np.isnan(row[temp_col]) else 24.0 - z * 0.015
                s_val = float(row[sal_col]) if sal_col and not np.isnan(row[sal_col]) else 34.5 + z * 0.001
                o_val = float(row[oxy_col]) if oxy_col and not np.isnan(row[oxy_col]) else max(10.0, 180.0 - z * 0.2)
                c_val = float(row[cur_col]) if cur_col and not np.isnan(row[cur_col]) else max(0.05, 0.45 - z * 0.0003)
                d_val = float(row[dir_col]) if dir_col and not np.isnan(row[dir_col]) else 90.0

                profile_points.append(ProfilePoint(
                    depth=round(z, 1),
                    temperature=round(t_val, 2),
                    salinity=round(s_val, 2),
                    currentSpeed=round(c_val, 2),
                    currentDir=round(d_val, 1),
                    oxygen=round(o_val, 1)
                ))

            z_max_group = max([p.depth for p in profile_points]) if profile_points else 1000.0
            parking_d = z_max_group * 0.5 if z_max_group > 50 else z_max_group

            floats.append(FloatRecord(
                id=clean_id,
                platform=f"In-situ Observation ({filename})",
                lat=round(flat, 4),
                lon=round(flon, 4),
                cycle=cycle_val,
                lastReportOffset=-12.0,
                parkingDepth=round(parking_d, 1),
                profile=ProfileResult(zmax=round(z_max_group, 1), points=profile_points),
                source=f"Tabular ingest · {filename} ({len(profile_points)} depth levels)"
            ))

            float_idx += 1
            if len(floats) >= 50:  # Cap at 50 profiles for UI performance
                break

        bbox = BoundingBox(
            min_lat=round(min_lat, 4),
            max_lat=round(max_lat, 4),
            min_lon=round(min_lon, 4),
            max_lon=round(max_lon, 4)
        )

        clean_id = f"custom_tab_{abs(hash(filename)) % 1000000}"
        clean_name = filename.replace(".csv", "").replace(".txt", "").replace("_", " ").title()

        # Compute summary physics
        t_mean = float(df[temp_col].mean()) if temp_col and not df[temp_col].isna().all() else 26.0
        s_mean = float(df[sal_col].mean()) if sal_col and not df[sal_col].isna().all() else 34.8
        o_mean = float(df[oxy_col].mean()) if oxy_col and not df[oxy_col].isna().all() else 160.0

        site_record = SitePhysics(
            id=clean_id,
            name=f"{clean_name} (Observations)",
            region=f"Coverage: {bbox.min_lat:.2f}° to {bbox.max_lat:.2f}°N, {bbox.min_lon:.2f}° to {bbox.max_lon:.2f}°E",
            lat=round(center_lat, 4),
            lon=round(center_lon, 4),
            maxDepth=max(50.0, round(max_depth, 1)),
            blurb=f"In-situ observation dataset '{filename}' parsed with {len(floats)} float/station profiles and {len(df)} depth measurements.",
            ts=round(t_mean + 2.0, 1),
            ss=round(s_mean - 0.5, 2),
            td=round(max(1.5, t_mean - 18.0), 1),
            sd=round(s_mean + 0.3, 2),
            mld=min(60.0, max(25.0, max_depth * 0.08)),
            tw=40.0,
            salMaxAmp=0.3,
            salMaxZ=110.0,
            flow=0.4,
            eddy=200.0,
            bgU=0.15,
            bgV=0.08,
            o2s=round(o_mean + 20.0, 1),
            o2d=round(max(30.0, o_mean - 40.0), 1),
            o2z0=85.0,
            o2minAmp=100.0,
            o2minZ=min(350.0, max_depth * 0.35),
            o2minW=180.0,
            bbox=bbox,
            variables=["temp", "sal", "cur", "oxy"],
            isCustom=True,
            sourceType="TABULAR_OBSERVATION",
            floats=floats
        )

        return {
            "site_id": clean_id,
            "filename": filename,
            "bbox": bbox,
            "depth_range": {"min": 0.0, "max": max_depth},
            "float_count": len(floats),
            "floats": floats,
            "site_record": site_record,
            "custom_observation": cls._extract_custom_observation(
                df, time_col, lat_col, lon_col,
                temp_col, sal_col, cur_u_col, cur_v_col, ssh_col, mld_col
            ),
        }

    @classmethod
    def _extract_custom_observation(
        cls,
        df: pd.DataFrame,
        time_col: Optional[str],
        lat_col: Optional[str],
        lon_col: Optional[str],
        temp_col: Optional[str],
        sal_col: Optional[str],
        cur_u_col: Optional[str],
        cur_v_col: Optional[str],
        ssh_col: Optional[str],
        mld_col: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts a single-row custom observation if the CSV contains Copernicus physical columns.
        Normalizes date to YYYY-MM-DD format preserving the exact date without substitution.
        """
        if df.empty:
            return None

        has_phys = any(col is not None for col in [temp_col, sal_col, cur_u_col, cur_v_col, ssh_col, mld_col])
        if not has_phys:
            return None

        row = df.iloc[0]

        obs_date = None
        if time_col and time_col in df.columns:
            raw_date = str(row[time_col]).strip()
            try:
                dt = pd.to_datetime(raw_date)
                obs_date = dt.strftime("%Y-%m-%d")
            except Exception:
                obs_date = raw_date[:10]

        obs: Dict[str, Any] = {}
        if obs_date:
            obs["date"] = obs_date
        if lat_col and lat_col in df.columns and pd.notna(row[lat_col]):
            obs["lat"] = float(row[lat_col])
        if lon_col and lon_col in df.columns and pd.notna(row[lon_col]):
            obs["lon"] = float(row[lon_col])

        if temp_col and temp_col in df.columns and pd.notna(row[temp_col]):
            obs["thetao"] = float(row[temp_col])
        if sal_col and sal_col in df.columns and pd.notna(row[sal_col]):
            obs["so"] = float(row[sal_col])
        if cur_u_col and cur_u_col in df.columns and pd.notna(row[cur_u_col]):
            obs["uo"] = float(row[cur_u_col])
        if cur_v_col and cur_v_col in df.columns and pd.notna(row[cur_v_col]):
            obs["vo"] = float(row[cur_v_col])
        if ssh_col and ssh_col in df.columns and pd.notna(row[ssh_col]):
            obs["zos"] = float(row[ssh_col])
        if mld_col and mld_col in df.columns and pd.notna(row[mld_col]):
            obs["mlotst"] = float(row[mld_col])

        return obs if len(obs) >= 3 else None

    @classmethod
    def to_geojson(cls, floats: List[FloatRecord]) -> Dict[str, Any]:
        """
        Converts parsed float records to OGC-compliant GeoJSON FeatureCollection.
        """
        features = []
        for f in floats:
            features.append({
                "type": "Feature",
                "id": f.id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [f.lon, f.lat, -f.parkingDepth]
                },
                "properties": {
                    "id": f.id,
                    "platform": f.platform,
                    "cycle": f.cycle,
                    "parkingDepth": f.parkingDepth,
                    "source": f.source,
                    "lastReportOffset": f.lastReportOffset,
                    "profile_depth_count": len(f.profile.points),
                    "profile": [p.model_dump() for p in f.profile.points]
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }
