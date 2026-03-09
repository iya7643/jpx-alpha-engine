from __future__ import annotations

import logging

from .config import load_settings
from .pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)


def main() -> None:
    settings = load_settings()
    run_pipeline(settings)


if __name__ == '__main__':
    main()
