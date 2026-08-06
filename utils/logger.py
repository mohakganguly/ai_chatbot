"""
Central logging configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "chatbot.log"


logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",

    handlers=[

        logging.FileHandler(LOG_FILE),

        logging.StreamHandler()

    ]

)


def get_logger(name: str):

    return logging.getLogger(name)