from __future__ import annotations

import logging
import argparse
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if __package__ in (None, "") and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __package__ in (None, ""):
    from runtime.udp_factory import UdpHostLifecycleServer
else:
    from .udp_factory import UdpHostLifecycleServer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one host lifecycle service.")
    parser.add_argument("--host", type=str, required=False)
    parser.add_argument("--port", type=int, required=False, default=10000)
    parser.add_argument("--physical-system-id", required=True)
    return parser.parse_args()

def configure_logging(log_name: str) -> None:
      log_dir = Path("logs")
      log_dir.mkdir(exist_ok=True)

      logging.basicConfig(
          level=logging.INFO,
          format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
          handlers=[
              logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / f"{log_name}.log", mode="a", encoding="utf-8"),
          ],
          force=True,
      )

def main() -> None:
    args = parse_args()
    configure_logging(f"lifecycle_args_{args.physical_system_id}")
    server = UdpHostLifecycleServer(hostifx=args.host, port=args.port,
        physical_system_id=args.physical_system_id)
    logger.info(f"Launching server on {args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.close()
    logger.info(f"Terminated server on {args.host}:{args.port}")

if __name__ == "__main__":
    main()
