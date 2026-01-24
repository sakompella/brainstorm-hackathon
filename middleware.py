#!/usr/bin/env python3
import asyncio

from middleware.config import Config
from middleware.pipeline import run_pipeline

if __name__ == "__main__":
    cfg = Config()
    try:
        asyncio.run(run_pipeline(cfg))
    except KeyboardInterrupt:
        print("\n[middleware] stopped.")
