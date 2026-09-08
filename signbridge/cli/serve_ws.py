"""Run the MEC-side WebSocket bridge, for the browser web UI.

    python -m signbridge.cli.serve_ws --config config.yaml

Same model, same checkpoint, same smoothing settings as
``signbridge.cli.serve`` - this just opens a second front door a browser can
actually connect to (WebSocket + JSON), since a browser cannot open the raw
TCP socket the capture host uses. Run this instead of, or alongside,
``signbridge.cli.serve`` depending on whether your capture device is the
Python host, a browser, or both at once (each front door tracks its own
per-connection sessions, so running both together is safe).

Before you have trained anything, ``--smoke-test`` behaves exactly as it does
for the TCP server: an untrained, randomly initialised model, so the browser
link can be checked end to end before any training.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..config import load_config
from ..server.predictor import TorchPredictor, UntrainedPredictor
from ..server.session import session_config_from
from ..server.ws_server import WebSocketBridge, run_forever

logger = logging.getLogger(__name__)

DEFAULT_WS_PORT = 8765


def main(argv: Optional[List[str]] = None) -> int:
    """Start the WebSocket bridge and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to serve (default: from config)")
    parser.add_argument("--host", default=None, help="Bind address (default: config's server.host)")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT, help=f"WebSocket port (default: {DEFAULT_WS_PORT})")
    parser.add_argument("--device", default=None, help="Force a device, e.g. cpu, mps, cuda")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Serve an untrained model to verify the browser link before any training",
    )
    parser.add_argument(
        "--smoke-labels",
        default="hello,thanks,yes,no,iloveyou",
        help="Placeholder labels for --smoke-test",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    try:
        import websockets  # noqa: F401
    except ImportError:
        print("\nThe 'websockets' package is required for the browser bridge.", file=sys.stderr)
        print("Install it with: pip install websockets\n", file=sys.stderr)
        return 1

    overrides = {"server": {"checkpoint": args.checkpoint, "host": args.host, "device": args.device}}
    config = load_config(args.config if Path(args.config).exists() else None, overrides=overrides)

    if args.smoke_test:
        labels = [item.strip() for item in args.smoke_labels.split(",") if item.strip()]
        predictor = UntrainedPredictor(config.feature_extractor(), labels, config.model.window)
        print("\n*** SMOKE TEST: the model is untrained and its predictions are")
        print("*** meaningless. This mode exists only to verify the browser link.\n")
    else:
        try:
            predictor = TorchPredictor(config.server.checkpoint, device=config.server.device)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            print("Or verify the link without a model:", file=sys.stderr)
            print("    python -m signbridge.cli.serve_ws --smoke-test", file=sys.stderr)
            return 1

    session_config = session_config_from(config.server)
    bridge = WebSocketBridge(predictor, session_config)

    info = predictor.info()
    print(f"SignBridge WebSocket bridge listening on ws://{config.server.host}:{args.port}")
    print(f"  labels ({len(info['labels'])}): {', '.join(info['labels'])}")
    print(f"  layout: {info['layout']['name']} ({info['layout']['num_landmarks']} landmarks)")
    print(f"  features: {info['feature_dim']} per frame, window {info['window']} frames")
    if "device" in info:
        print(f"  device: {info['device']}")
    print("\nOpen web/index.html in a browser and point it at this address. Ctrl-C to stop.\n")
    sys.stdout.flush()

    try:
        run_forever(predictor, session_config, config.server.host, args.port)
    except KeyboardInterrupt:
        pass
    print(f"\nStopping. {bridge.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
