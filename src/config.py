"""
Store config data for all data_prep files centrally
"""

from dataclasses import dataclass


@dataclass
class SceneConfig:
    max_scene_size: int = 3000
    min_paragraph_size: int = 75
    debug_mode: bool = True
    debug_dir: str = "./data/debug/"
