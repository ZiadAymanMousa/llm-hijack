"""Load an experiment config (configs/*.yaml)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def results_dir(self) -> Path:
        return ROOT / self.raw["results_dir"]

    def model(self, name: str) -> dict:
        return self.raw["models"][name]

    def arm(self, name: str) -> dict:
        return self.raw["arms"][name]

    def agent(self, name: str) -> dict:
        return self.raw["agents"][name]

    def guard(self, name: str) -> dict:
        return self.raw["guards"][name]

    def trusted_constants(self, suite: str) -> list[str]:
        return self.raw.get("trusted_constants", {}).get(suite, [])


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return Config(yaml.safe_load(f), path)
