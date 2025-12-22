from __future__ import annotations

import argparse

from .runner import ConfigError, run_from_config

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srvar")
    parser.add_argument(
        "--version",
        action="version",
        version=f"srvar {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    validate_p = sub.add_parser("validate", help="Validate a YAML config file")
    validate_p.add_argument("config", type=str, help="Path to config.yml")

    run_p = sub.add_parser("run", help="Run fit/forecast from a YAML config file")
    run_p.add_argument("config", type=str, help="Path to config.yml")
    run_p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output directory (also copies config.yml there)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return 0

    try:
        if args.command == "validate":
            run_from_config(args.config, validate_only=True)
            return 0
        if args.command == "run":
            run_from_config(args.config, out_dir=args.out, validate_only=False)
            return 0
        raise ValueError(f"unknown command: {args.command}")
    except ConfigError as e:
        parser.error(str(e))
        return 2
