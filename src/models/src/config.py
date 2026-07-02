# src/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 50
    image_height: int = 512
    image_width: int = 512

def load_config(path: str) -> TrainingConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    expected_fields = {"learning_rate", "batch_size", "num_epochs", "image_height", "image_width"}
    missing = expected_fields - data.keys()
    if missing:
        logger.warning(f"Missing fields in config, using defaults: {missing}")

    cfg = TrainingConfig(**data)
    logger.debug(f"Raw YAML data: {data}")
    return cfg

if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = load_config(str(config_path))
    logger.info(f"Config loaded: {cfg}")