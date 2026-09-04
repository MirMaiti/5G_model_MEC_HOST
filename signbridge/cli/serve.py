"""Run the MEC-side TCP server.

    python -m signbridge.cli.serve --config config.yaml

Loads a checkpoint and listens for capture hosts. This process never opens a
camera and never imports MediaPipe - it only ever sees landmark coordinates.

Before you have trained anything, ``--smoke-test`` starts an untrained,
randomly initialised model so the network path can be checked end to end. Its
predictions are meaningless and it says so in the handshake; the host will
refuse to speak them.
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
from ..server.tcp_server import SignBridgeServer

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """Start the server and serve until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to serve (default: from config)")
    parser.add_argument("--host", default=None, help="Bind address override")
    parser.add_argument("--port", type=int, default=None, help="Bind port override")
    parser.add_argument("--device", default=None, help="Force a device, e.g. cpu, mps, cuda")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Serve an untrained model to verify the link before any training",
    )
    parser.add_argument(
        "--smoke-labels",
        default="hello,thanks,yes,no,iloveyou",
        help="Placeholder labels for --smoke-test",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    overrides = {
        "server": {
            "checkpoint": args.checkpoint,
            "host": args.host,
            "port": args.port,
            "device": args.device,
        }
    }
    config = load_config(args.config if Path(args.config).exists() else None, overrides=overrides)

    if args.smoke_test:
        labels = [item.strip() for item in args.smoke_labels.split(",") if item.strip()]
        predictor = UntrainedPredictor(config.feature_extractor(), labels, config.model.window)
        print("\n*** SMOKE TEST: the model is untrained and its predictions are")
        print("*** meaningless. This mode exists only to verify the TCP path.\n")
    else:
        try:
            predictor = TorchPredictor(config.server.checkpoint, device=config.server.device)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            print("Or verify the link without a model:", file=sys.stderr)
            print("    python -m signbridge.cli.serve --smoke-test", file=sys.stderr)
            return 1

    server = SignBridgeServer(
        address=(config.server.host, config.server.port),
        predictor=predictor,
        session_config=session_config_from(config.server),
        max_connections=config.server.max_connections,
    )

    info = predictor.info()
    print(f"SignBridge server listening on {config.server.host}:{server.port}")
    print(f"  labels ({len(info['labels'])}): {', '.join(info['labels'])}")
    print(f"  layout: {info['layout']['name']} ({info['layout']['num_landmarks']} landmarks)")
    print(f"  features: {info['feature_dim']} per frame, window {info['window']} frames")
    if "device" in info:
        print(f"  device: {info['device']}")
    print("\nCtrl-C to stop.\n")
    # Redirected stdout is block-buffered; flush so an operator tailing a log
    # file sees what the server is serving before the first host connects.
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping. {server.stats()}")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
