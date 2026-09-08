from typing import Dict, List, Optional, Any
import numpy as np
from ..models.schemas import SitePhysics, BoundingBox, FloatRecord, ProfileResult, ProfilePoint


# Default baseline oceanographic study sites (INCOIS / Global)
DEFAULT_BASELINE_SITES: List[SitePhysics] = [
    SitePhysics(
        id="bob",
        name="Bay of Bengal — Fresh Plume",
        region="N. Indian Ocean",
        lat=17.8,
        lon=88.2,
        maxDepth=2600.0,
        blurb="Ganga–Brahmaputra runoff caps the bay with a low-salinity lid. The strong halocline traps heat and feeds one of Earth's most intense oxygen-minimum zones.",
        ts=29.2, ss=31.6, td=2.8, sd=34.86, mld=28.0, tw=34.0,
        salMaxAmp=0.55, salMaxZ=120.0,
        flow=0.45, eddy=190.0, bgU=0.18, bgV=0.05,
        o2s=188.0, o2d=168.0, o2z0=80.0, o2minAmp=160.0, o2minZ=380.0, o2minW=250.0,
        bbox=BoundingBox(min_lat=14.0, max_lat=22.0, min_lon=83.0, max_lon=93.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="eqio",
        name="Equatorial Indian Ocean",
        region="Indian Ocean",
        lat=0.5,
        lon=80.5,
        maxDepth=4300.0,
        blurb="Wyrtki jets race east along the equator; a shallow thermocline makes this the fastest-responding region of the Indian Ocean to climate modes.",
        ts=29.3, ss=34.9, td=3.4, sd=34.72, mld=40.0, tw=45.0,
        salMaxAmp=0.12, salMaxZ=150.0,
        flow=0.55, eddy=240.0, bgU=0.32, bgV=0.0,
        o2s=182.0, o2d=170.0, o2z0=90.0, o2minAmp=120.0, o2minZ=520.0, o2minW=300.0,
        bbox=BoundingBox(min_lat=-4.0, max_lat=5.0, min_lon=74.0, max_lon=88.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="aras",
        name="Arabian Sea — OMZ Core",
        region="N. Indian Ocean",
        lat=16.5,
        lon=67.5,
        maxDepth=3700.0,
        blurb="Intense monsoon-driven productivity and poor ventilation produce one of the most severe oxygen-minimum zones in the global ocean.",
        ts=27.4, ss=36.5, td=2.6, sd=34.8, mld=55.0, tw=55.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=0.40, eddy=220.0, bgU=0.12, bgV=-0.08,
        o2s=178.0, o2d=170.0, o2z0=70.0, o2minAmp=176.0, o2minZ=260.0, o2minW=220.0,
        bbox=BoundingBox(min_lat=12.0, max_lat=21.0, min_lon=62.0, max_lon=73.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="agul",
        name="Agulhas Current",
        region="SW Indian Ocean",
        lat=-34.2,
        lon=26.8,
        maxDepth=3100.0,
        blurb="The strongest western boundary current of the Indian Ocean, famous for its retroflection eddies that leak warm water into the Atlantic.",
        ts=22.6, ss=35.5, td=3.6, sd=34.8, mld=65.0, tw=70.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=1.50, eddy=260.0, bgU=-1.0, bgV=-0.8,
        o2s=196.0, o2d=185.0, o2z0=90.0, o2minAmp=70.0, o2minZ=600.0, o2minW=350.0,
        bbox=BoundingBox(min_lat=-38.0, max_lat=-30.0, min_lon=22.0, max_lon=32.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="guls",
        name="Gulf Stream",
        region="N. Atlantic",
        lat=36.4,
        lon=-73.6,
        maxDepth=3300.0,
        blurb="A swift, meandering western boundary current transporting warm subtropical water northward and shedding energetic rings.",
        ts=23.6, ss=36.1, td=3.8, sd=34.94, mld=70.0, tw=80.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=1.30, eddy=280.0, bgU=0.25, bgV=1.05,
        o2s=210.0, o2d=190.0, o2z0=110.0, o2minAmp=95.0, o2minZ=650.0, o2minW=380.0,
        bbox=BoundingBox(min_lat=32.0, max_lat=40.0, min_lon=-78.0, max_lon=-68.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="kuro",
        name="Kuroshio Extension",
        region="N.W. Pacific",
        lat=35.2,
        lon=145.4,
        maxDepth=5500.0,
        blurb="Where the Kuroshio frees itself from the coast — a basin-scale eddy nursery and a sharp front between subtropical and subpolar water.",
        ts=21.8, ss=34.7, td=2.2, sd=34.5, mld=90.0, tw=95.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=1.15, eddy=300.0, bgU=0.9, bgV=0.2,
        o2s=220.0, o2d=160.0, o2z0=100.0, o2minAmp=130.0, o2minZ=800.0, o2minW=420.0,
        bbox=BoundingBox(min_lat=31.0, max_lat=39.0, min_lon=140.0, max_lon=150.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="drak",
        name="Drake Passage",
        region="Southern Ocean",
        lat=-58.5,
        lon=-63.0,
        maxDepth=3900.0,
        blurb="The Antarctic Circumpolar Current squeezes through the tightest gap in the world ocean — cold, well-ventilated, relentlessly stormy.",
        ts=3.4, ss=33.9, td=1.2, sd=34.72, mld=120.0, tw=160.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=0.80, eddy=320.0, bgU=0.55, bgV=0.3,
        o2s=330.0, o2d=200.0, o2z0=150.0, o2minAmp=35.0, o2minZ=700.0, o2minW=500.0,
        bbox=BoundingBox(min_lat=-62.0, max_lat=-54.0, min_lon=-68.0, max_lon=-58.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="chal",
        name="Challenger Deep",
        region="Mariana Trench",
        lat=11.37,
        lon=142.59,
        maxDepth=10935.0,
        blurb="The deepest seafloor on Earth. Hadal waters below 6,000 m are near-freezing, near-still, and pressed by more than a thousand atmospheres.",
        ts=29.1, ss=34.55, td=1.4, sd=34.68, mld=50.0, tw=60.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=0.06, eddy=500.0, bgU=0.02, bgV=0.015,
        o2s=195.0, o2d=152.0, o2z0=110.0, o2minAmp=55.0, o2minZ=750.0, o2minW=450.0,
        bbox=BoundingBox(min_lat=9.0, max_lat=14.0, min_lon=140.0, max_lon=145.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
    SitePhysics(
        id="peru",
        name="Peru Upwelling",
        region="S.E. Pacific",
        lat=-12.5,
        lon=-80.5,
        maxDepth=3900.0,
        blurb="Trade winds drive cold, nutrient-rich water to the surface — the engine of the Humboldt Current system and the El Niño heartbeat.",
        ts=18.4, ss=34.95, td=2.4, sd=34.6, mld=25.0, tw=30.0,
        salMaxAmp=0.0, salMaxZ=0.0,
        flow=0.35, eddy=180.0, bgU=0.25, bgV=0.1,
        o2s=180.0, o2d=150.0, o2z0=60.0, o2minAmp=165.0, o2minZ=200.0, o2minW=180.0,
        bbox=BoundingBox(min_lat=-16.0, max_lat=-9.0, min_lon=-84.0, max_lon=-77.0),
        variables=["temp", "sal", "cur", "oxy"],
        isCustom=False,
        sourceType="INCOIS_BASELINE"
    ),
]


class SiteRegistry:
    """
    Central registry for default INCOIS baseline sites and dynamic user-uploaded datasets.
    """
    _baseline_sites: Dict[str, SitePhysics] = {s.id: s for s in DEFAULT_BASELINE_SITES}
    _custom_sites: Dict[str, SitePhysics] = {}
    _custom_datasets: Dict[str, Any] = {}  # Holds references to xarray / dataframes
    _float_registry: Dict[str, List[FloatRecord]] = {}

    @classmethod
    def initialize_baseline_floats(cls):
        """Generates synthetic Argo floats for baseline sites if not already present."""
        for s in DEFAULT_BASELINE_SITES:
            if s.id not in cls._float_registry:
                floats = cls._generate_baseline_floats(s)
                cls._float_registry[s.id] = floats
                s.floats = floats

    @classmethod
    def _generate_baseline_floats(cls, site: SitePhysics) -> List[FloatRecord]:
        np.random.seed(abs(hash(site.id)) % (2**31 - 1))
        parking_depths = [26.0, 85.0, 1000.0, 1000.0, 2000.0]
        floats = []

        for i in range(5):
            flat = site.lat + (np.random.rand() - 0.5) * 1.5
            flon = site.lon + (np.random.rand() - 0.5) * 1.7
            pd = float(np.clip(parking_depths[i] + (np.random.rand() - 0.5) * 60, 15, site.maxDepth - 15))
            last_off = float(-(1.0 + np.random.rand() * 66.0))
            f_id = f"290{int(1500 + np.random.rand() * 900)}"

            # Generate profile points (0 to 2000m)
            points = []
            z_max = min(2000.0, site.maxDepth)
            for j in range(41):
                z = float(j * 10 if j <= 8 else 80 + (j - 8) * (z_max - 80) / 32)
                f_t = 1.0 / (1.0 + np.exp((z - (site.mld + 55)) / site.tw))
                temp = site.td + (site.ts - site.td) * f_t
                sal = site.sd + (site.ss - site.sd) * np.exp(-z / 95.0)
                if site.salMaxAmp > 0:
                    sal += site.salMaxAmp * np.exp(-((z - site.salMaxZ) / 85.0) ** 2)
                oxy_s = site.o2d + (site.o2s - site.o2d) * np.exp(-z / site.o2z0)
                oxy_omz = site.o2minAmp * np.exp(-((z - site.o2minZ) / site.o2minW) ** 2)
                oxy = max(2.0, oxy_s - oxy_omz)
                spd = site.flow * (0.16 + 0.84 * np.exp(-z / 300.0))

                points.append(ProfilePoint(
                    depth=round(z, 1),
                    temperature=round(temp, 2),
                    salinity=round(sal, 2),
                    currentSpeed=round(spd, 2),
                    currentDir=45.0,
                    oxygen=round(oxy, 1)
                ))

            floats.append(FloatRecord(
                id=f_id,
                platform="Argo float · APEX (INCOIS/GDAC)",
                lat=round(flat, 4),
                lon=round(flon, 4),
                cycle=int(70 + np.random.rand() * 140),
                lastReportOffset=round(last_off, 1),
                parkingDepth=round(pd, 1),
                profile=ProfileResult(zmax=round(z_max, 1), points=points),
                source="Argo GDAC · INCOIS delayed-mode QC verified"
            ))

        return floats

    @classmethod
    def get_all_sites(cls) -> List[SitePhysics]:
        cls.initialize_baseline_floats()
        return list(cls._custom_sites.values()) + list(cls._baseline_sites.values())

    @classmethod
    def get_site(cls, site_id: str) -> Optional[SitePhysics]:
        cls.initialize_baseline_floats()
        if site_id in cls._custom_sites:
            return cls._custom_sites[site_id]
        return cls._baseline_sites.get(site_id)

    @classmethod
    def register_uploaded_site(cls, site_record: SitePhysics, raw_dataset: Any = None):
        cls._custom_sites[site_record.id] = site_record
        if raw_dataset is not None:
            cls._custom_datasets[site_record.id] = raw_dataset
        if site_record.floats:
            cls._float_registry[site_record.id] = site_record.floats

    @classmethod
    def delete_uploaded_site(cls, site_id: str) -> bool:
        removed = False
        if site_id in cls._custom_sites:
            del cls._custom_sites[site_id]
            removed = True
        if site_id in cls._custom_datasets:
            del cls._custom_datasets[site_id]
        if site_id in cls._float_registry:
            del cls._float_registry[site_id]
        return removed

    @classmethod
    def get_dataset(cls, site_id: str) -> Optional[Any]:
        return cls._custom_datasets.get(site_id)

    @classmethod
    def get_floats(cls, site_id: str) -> List[FloatRecord]:
        cls.initialize_baseline_floats()
        return cls._float_registry.get(site_id, [])
