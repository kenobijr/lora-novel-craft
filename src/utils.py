import argparse


def parse_scene_range(value: str) -> tuple[int, int]:
    """
    - serveral cli entry points receive scene range with start end values
    - parse 'start,end' string into tuple for argparse
    """
    try:
        parts = value.split(",")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(f"must be start,end format (e.g. 0,10), got: {value}")
