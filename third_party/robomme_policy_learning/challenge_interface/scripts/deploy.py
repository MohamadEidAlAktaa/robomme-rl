"""
This is used by participants to serve their policy. 

Participants may need to modify this file to adapt to their own policy. for example, loading multiple model ckpts.

Provide an example usage here if using docker submission:

uv run python -m  challenge_interface.scripts.deploy --checkpoint-dir perceptual-framesamp-modul/79999
"""

import argparse
from mme_vla_suite.policies.policy_config import create_trained_policy
from pathlib import Path
from mme_vla_suite.training.config import get_config


from challenge_interface.server import PolicyServer
from challenge_interface.server_http import PolicyHTTPServer
from challenge_interface.policy import MyPolicy_for_CVPR_Challenge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a policy for the CVPR challenge.")
    parser.add_argument(
        "--transport",
        type=str,
        choices=("websocket", "http"),
        default="websocket",
        help="Server transport to use (default: %(default)s).",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the policy server (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("my_cool_model"),
        help="Path to the checkpoint directory (default: %(default)s).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = create_trained_policy(
        train_config=get_config("mme_vla_suite"),
        checkpoint_dir=args.checkpoint_dir,
        seed=7,
    )

    policy = MyPolicy_for_CVPR_Challenge(model=model)
    if args.transport == "http":
        policy_server = PolicyHTTPServer(policy, host=args.host, port=args.port)
    else:
        policy_server = PolicyServer(policy, host=args.host, port=args.port)
    policy_server.serve_forever()


if __name__ == "__main__":
    main()