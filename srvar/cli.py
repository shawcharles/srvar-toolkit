from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

from .runner import ConfigError, backtest_from_config, run_from_config

from . import __version__


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        f /= 1024.0
        if f < 1024.0:
            return f"{f:.1f} {u}"
    return f"{f:.1f} PiB"


def _supports_color(no_color: bool) -> bool:
    if no_color:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class _Reporter:
    def __init__(self, *, color: bool, verbose: bool) -> None:
        self._color = color
        self._verbose = verbose
        self._stage_index = 0
        self._out_dir: Path | None = None
        self._summaries: dict[str, dict[str, Any]] = {}
        self._artifacts: list[dict[str, Any]] = []

    def _c(self, code: str, s: str) -> str:
        if not self._color:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    def _b(self, s: str) -> str:
        return self._c("1", s)

    def _dim(self, s: str) -> str:
        return self._c("2", s)

    def _ok(self, s: str) -> str:
        return self._c("32", s)

    def _info(self, s: str) -> str:
        return self._c("36", s)

    def _warn(self, s: str) -> str:
        return self._c("33", s)

    def header(self, *, command: str, config: str) -> None:
        width = shutil.get_terminal_size(fallback=(88, 24)).columns
        line = "=" * min(width, 88)
        print(self._b(f"srvar-toolkit {__version__}"))
        print(self._dim(line))
        print(f"{self._b('Command')}: {command}")
        print(f"{self._b('Config')}:   {config}")
        print(self._dim(line))

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "stage_start":
            name = str(payload.get("name"))
            self._stage_index += 1
            print(f"{self._info('[..]')} {self._b(name)}")
            return

        if event == "backtest_origin":
            if self._verbose:
                i = int(payload.get("i", 0))
                k = int(payload.get("k", 0))
                origin_end = payload.get("origin_end")
                train_T = payload.get("train_T")
                elapsed_s = float(payload.get("elapsed_s", 0.0))
                print(
                    f"{self._dim('  ->')} origin {i+1}/{k}: end={origin_end}  train_T={train_T} {self._dim(f'({elapsed_s:.3f}s)')}"
                )
            return

        if event == "stage_end":
            name = str(payload.get("name"))
            elapsed_s = float(payload.get("elapsed_s", 0.0))
            print(f"{self._ok('[OK]')} {self._b(name)} {self._dim(f'({elapsed_s:.3f}s)')}")
            return

        if event == "summary":
            kind = str(payload.get("kind"))
            self._summaries[kind] = payload

            if kind == "dataset":
                vars_s = ",".join([str(v) for v in payload.get("variables", [])])
                start = payload.get("start")
                end = payload.get("end")
                span = ""
                if isinstance(start, str) and isinstance(end, str) and start and end:
                    span = f"  span=[{start} .. {end}]"
                print(
                    f"{self._b('  dataset')}: T={payload.get('T')}  N={payload.get('N')}  vars=[{vars_s}]{span}"
                )
                return

            if kind == "model":
                elb = "on" if payload.get("elb") else "off"
                sv = "on" if payload.get("sv") else "off"
                intercept = "yes" if payload.get("include_intercept") else "no"
                print(
                    f"{self._b('  model')}:   p={payload.get('p')}  intercept={intercept}  ELB={elb}  SV={sv}"
                )
                return

            if kind == "prior":
                print(
                    f"{self._b('  prior')}:   family={payload.get('family')}  method={payload.get('method')}"
                )
                return

            if kind == "sampler":
                print(
                    f"{self._b('  sampler')}: draws={payload.get('draws')}  burn_in={payload.get('burn_in')}  thin={payload.get('thin')}"
                )
                return

            if kind == "forecast":
                print(
                    f"{self._b('  forecast')}: horizons={payload.get('horizons')}  draws={payload.get('draws')}  q={payload.get('quantile_levels')}"
                )
                return

            if kind == "output":
                out_dir = payload.get("out_dir")
                if isinstance(out_dir, str):
                    self._out_dir = Path(out_dir)
                print(
                    f"{self._b('  output')}:  dir={payload.get('out_dir')}  save_fit={payload.get('save_fit')}  save_forecast={payload.get('save_forecast')}  save_plots={payload.get('save_plots')}"
                )
                return

            if kind == "backtest":
                print(
                    f"{self._b('  backtest')}: mode={payload.get('mode')}  origins={payload.get('origins')}  horizons={payload.get('horizons')}  draws={payload.get('draws')}"
                )
                return

            if self._verbose:
                print(f"{self._warn('  summary')}: {kind}={payload}")
            return

        if event == "artifact":
            self._artifacts.append(payload)
            if self._verbose:
                p = str(payload.get("path"))
                b = int(payload.get("bytes", 0))
                k = str(payload.get("kind"))
                print(f"{self._dim('  ->')} {k}: {p} {self._dim(_human_bytes(b))}")
            return

        if event == "run_end":
            elapsed_s = float(payload.get("elapsed_s", 0.0))
            width = shutil.get_terminal_size(fallback=(88, 24)).columns
            line = "-" * min(width, 88)
            print(self._dim(line))
            print(f"{self._ok('Run complete')} {self._dim(f'({elapsed_s:.3f}s total)')}")

            if self._out_dir is not None:
                out_path = self._out_dir.resolve()
                print(f"{self._b('Outputs')}: {out_path}")

            if self._artifacts:
                print(self._b("Artifacts:"))
                out_dir_res = self._out_dir.resolve() if self._out_dir is not None else None
                rows: list[tuple[str, str, str]] = []
                for a in self._artifacts:
                    kind = str(a.get("kind", ""))
                    path_s = str(a.get("path", ""))
                    size_s = _human_bytes(int(a.get("bytes", 0)))

                    display = path_s
                    if out_dir_res is not None:
                        try:
                            display = str(Path(path_s).resolve().relative_to(out_dir_res))
                        except Exception:
                            display = Path(path_s).name
                    rows.append((kind, display, size_s))

                w_kind = max(4, *(len(r[0]) for r in rows))
                w_file = max(4, *(len(r[1]) for r in rows))
                w_size = max(4, *(len(r[2]) for r in rows))
                w_file = min(w_file, 64)

                print(f"  {'KIND'.ljust(w_kind)}  {'FILE'.ljust(w_file)}  {'SIZE'.rjust(w_size)}")
                for kind, file_s, size_s in rows:
                    file_s2 = file_s
                    if len(file_s2) > w_file:
                        file_s2 = "…" + file_s2[-(w_file - 1) :]
                    print(f"  {kind.ljust(w_kind)}  {file_s2.ljust(w_file)}  {size_s.rjust(w_size)}")
            return

        if event == "validate_end":
            elapsed_s = float(payload.get("elapsed_s", 0.0))
            width = shutil.get_terminal_size(fallback=(88, 24)).columns
            line = "-" * min(width, 88)
            print(self._dim(line))
            print(f"{self._ok('Validation complete')} {self._dim(f'({elapsed_s:.3f}s total)')}")
            return

        if event == "backtest_end":
            elapsed_s = float(payload.get("elapsed_s", 0.0))
            width = shutil.get_terminal_size(fallback=(88, 24)).columns
            line = "-" * min(width, 88)
            print(self._dim(line))
            print(f"{self._ok('Backtest complete')} {self._dim(f'({elapsed_s:.3f}s total)')}")

            if self._out_dir is not None:
                out_path = self._out_dir.resolve()
                print(f"{self._b('Outputs')}: {out_path}")

            if self._artifacts:
                print(self._b("Artifacts:"))
                out_dir_res = self._out_dir.resolve() if self._out_dir is not None else None
                rows: list[tuple[str, str, str]] = []
                for a in self._artifacts:
                    kind = str(a.get("kind", ""))
                    path_s = str(a.get("path", ""))
                    size_s = _human_bytes(int(a.get("bytes", 0)))

                    display = path_s
                    if out_dir_res is not None:
                        try:
                            display = str(Path(path_s).resolve().relative_to(out_dir_res))
                        except Exception:
                            display = Path(path_s).name
                    rows.append((kind, display, size_s))

                w_kind = max(4, *(len(r[0]) for r in rows))
                w_file = max(4, *(len(r[1]) for r in rows))
                w_size = max(4, *(len(r[2]) for r in rows))
                w_file = min(w_file, 64)

                print(f"  {'KIND'.ljust(w_kind)}  {'FILE'.ljust(w_file)}  {'SIZE'.rjust(w_size)}")
                for kind, file_s, size_s in rows:
                    file_s2 = file_s
                    if len(file_s2) > w_file:
                        file_s2 = "…" + file_s2[-(w_file - 1) :]
                    print(f"  {kind.ljust(w_kind)}  {file_s2.ljust(w_file)}  {size_s.rjust(w_size)}")
            return


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
    validate_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    validate_p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in console output",
    )
    validate_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed progress output",
    )

    run_p = sub.add_parser("run", help="Run fit/forecast from a YAML config file")
    run_p.add_argument("config", type=str, help="Path to config.yml")
    run_p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output directory (also copies config.yml there)",
    )
    run_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    run_p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in console output",
    )
    run_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed progress output",
    )

    backtest_p = sub.add_parser("backtest", help="Run a rolling/expanding backtest from a YAML config file")
    backtest_p.add_argument("config", type=str, help="Path to config.yml")
    backtest_p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override output directory (also copies config.yml there)",
    )
    backtest_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    backtest_p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in console output",
    )
    backtest_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show more detailed progress output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return 0

    try:
        if args.command == "validate":
            reporter = None
            if not args.quiet:
                reporter = _Reporter(color=_supports_color(bool(args.no_color)), verbose=bool(args.verbose))
                reporter.header(command="srvar validate", config=str(args.config))
            run_from_config(args.config, validate_only=True, progress=reporter)
            if not args.quiet:
                print(f"{reporter._ok('Config OK')}: {args.config}" if reporter is not None else f"srvar: config OK: {args.config}")
            return 0
        if args.command == "run":
            reporter = None
            if not args.quiet:
                reporter = _Reporter(color=_supports_color(args.no_color), verbose=bool(args.verbose))
                reporter.header(command="srvar run", config=str(args.config))

            run_from_config(args.config, out_dir=args.out, validate_only=False, progress=reporter)
            return 0
        if args.command == "backtest":
            reporter = None
            if not args.quiet:
                reporter = _Reporter(color=_supports_color(bool(args.no_color)), verbose=bool(args.verbose))
                reporter.header(command="srvar backtest", config=str(args.config))

            backtest_from_config(args.config, out_dir=args.out, progress=reporter)
            return 0
        raise ValueError(f"unknown command: {args.command}")
    except ConfigError as e:
        parser.error(str(e))
        return 2
