from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(config: dict) -> None:
    log_dir = Path(config["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(config["runtime"]["log_level"]).upper(), logging.INFO)
    handler = RotatingFileHandler(log_dir / "inspection.log", maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler, console], force=True)
