"""Settings, region configuration and the shared domain types.

Regions, archetypes, thresholds, prices and storage are configuration, never code.
Moving to a new region is a new YAML file, not a new module.
"""

from __future__ import annotations

import functools
import pathlib

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    env: str = "dev"
    data_root: pathlib.Path = pathlib.Path("./data")
    artifact_root: pathlib.Path = pathlib.Path("./artifacts")
    replay_mode: bool = False
    log_level: str = "INFO"
    # A forecast older than this is refused rather than served as if current.
    # Generous by default: a stale number the caller can see the age of beats a
    # 503 during a demo. Replay mode bypasses it entirely -- frozen data is the
    # point there, not a fault.
    stale_after_minutes: int = 1440
    # Softer bar: /health warns here rather than erroring, so ingest lag is
    # visible long before it becomes a refusal.
    stale_warn_after_minutes: int = 180


class Archetype(BaseModel):
    """One representative fleet configuration within a region."""

    id: str
    share: float = Field(ge=0, le=1)
    # solar
    tilt: float | None = None
    azimuth: float | None = None
    tracking: str | None = None
    dc_ac_ratio: float | None = None
    gamma_pdc: float | None = None
    albedo: float | None = None
    soiling_pct_per_day: float | None = None
    # wind
    hub_height_m: float | None = None
    rotor_diameter_m: float | None = None
    cut_in_ms: float | None = None
    rated_ms: float | None = None
    cut_out_ms: float | None = None
    shear_alpha: float | None = None
    wake_loss_frac: float | None = None


class GridPoint(BaseModel):
    id: str
    lat: float
    lon: float
    weight: float = Field(ge=0, le=1)


class StorageConfig(BaseModel):
    """A virtual BESS. Declared, not measured -- which is what makes the
    capacity sweep in the planning screen possible."""

    energy_capacity_mwh: float = 400
    power_rating_mw: float = 100
    round_trip_efficiency: float = 0.88
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.95
    degradation_per_cycle: float = 3e-5
    response_time_min: int = 5


class Thresholds(BaseModel):
    ramp_limit_mw_per_h: float
    band_limit_frac: float = 0.35
    must_run_mw: float


class Prices(BaseModel):
    curtailment_opportunity_cost_per_mwh: float = 3000
    backup_fuel_cost_per_mwh: float = 6500


class RegionConfig(BaseModel):
    region_id: str
    timezone: str
    resolution_min: int = 15
    # A region with no metered, technology-separated generation series has
    # nothing to fit a residual model to and nothing to calibrate a band
    # against, so it is served from physics alone. Declared here rather than
    # inferred from a missing artifact: "no model file" must stay a loud error
    # for a region that is supposed to have one.
    physics_only: bool = False
    capacity_mw: dict[str, float]  # {"solar": ..., "wind": ...}
    weather_grid: list[GridPoint]
    nwp_models: list[str]
    archetypes: dict[str, list[Archetype]]
    decision_thresholds: Thresholds
    storage: StorageConfig
    prices: Prices

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> RegionConfig:
        for tech, arcs in self.archetypes.items():
            total = sum(a.share for a in arcs)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"{self.region_id}/{tech} archetype shares sum to {total}, expected 1.0"
                )
        return self

    @model_validator(mode="after")
    def _grid_weights_sum_to_one(self) -> RegionConfig:
        total = sum(p.weight for p in self.weather_grid)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{self.region_id} grid point weights sum to {total}, expected 1.0")
        return self


class SiteMaster(BaseModel):
    """One archetype resolved against its region: geometry + fleet parameters.

    This is what the physics chain consumes. `capacity_mw` is this archetype's
    share of the regional fleet, and it is the normaliser for everything
    downstream -- the model trains on capacity factor, not MW.
    """

    site_id: str
    region_id: str
    tech: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    timezone: str = "UTC"
    capacity_mw: float
    capacity_share: float
    archetype: Archetype


class GridState(BaseModel):
    """Operational state the decision engine needs beyond the forecast itself."""

    region_id: str
    must_run_mw: float
    storage_soc_mwh: float
    storage: StorageConfig
    ramp_limit_mw_per_h: float
    available_capacity_mw: float | None = None


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


REGION_DIR = pathlib.Path("config/regions")


@functools.lru_cache
def load_region(region_id: str) -> RegionConfig:
    """Load a region by its id, e.g. "BE".

    Filenames are human-facing (`belgium.yaml`, `india_rajasthan.yaml`) while ids
    are short codes, so match on the `region_id` field inside the file rather
    than forcing the filename to equal the id.
    """
    direct = REGION_DIR / f"{region_id.lower()}.yaml"
    candidates = [direct] if direct.exists() else sorted(REGION_DIR.glob("*.yaml"))
    for path in candidates:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if str(raw.get("region_id", "")).upper() == region_id.upper():
            return RegionConfig(**raw)
    known = sorted(
        str(yaml.safe_load(p.read_text(encoding="utf-8")).get("region_id", "?"))
        for p in REGION_DIR.glob("*.yaml")
    )
    raise FileNotFoundError(
        f"no config for region {region_id!r}. Configured regions: {known or 'none'}. "
        f"Add a YAML file under {REGION_DIR}/ with a matching region_id field."
    )


def site_master(cfg: RegionConfig, tech: str) -> list[SiteMaster]:
    """Expand a region's archetypes into the SiteMaster rows the physics chain runs on.

    The weather grid centroid is used as each archetype's location: a regional
    fleet has no single coordinate, and spatial variation is already carried by
    the capacity-weighted grid sampling in the feature layer.
    """
    lat = sum(p.lat * p.weight for p in cfg.weather_grid)
    lon = sum(p.lon * p.weight for p in cfg.weather_grid)
    total = cfg.capacity_mw[tech]
    return [
        SiteMaster(
            site_id=f"{cfg.region_id}-{tech.upper()}-{a.id.upper()}",
            region_id=cfg.region_id,
            tech=tech,
            lat=lat,
            lon=lon,
            timezone=cfg.timezone,
            capacity_mw=total * a.share,
            capacity_share=a.share,
            archetype=a,
        )
        for a in cfg.archetypes[tech]
    ]
