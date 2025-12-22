import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
except Exception:
    Console = None
    Panel = None
    Progress = None
    Table = None
    Text = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srvar import Dataset
from srvar.api import fit, forecast
from srvar.elb import ElbSpec
from srvar.sv import VolatilitySpec
from srvar.metrics import crps_draws
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.data.transformations import tcode_matrix
from srvar.data.vintages import dataset_from_vintage, load_vintages_from_dir


def _period(label: str) -> pd.Period:
    s = str(label).strip().replace(" ", "")
    return pd.Period(s, freq="Q")


def _seed_for(*, base_seed: int, model_id: int, origin: pd.Period, stage: int) -> int:
    x = int(base_seed)
    x = (x + 1000003 * int(model_id)) % 2**32
    x = (x + 1009 * int(origin.year) + 31 * int(origin.quarter)) % 2**32
    x = (x + 7919 * int(stage)) % 2**32
    return int(x)


def _trim_after_tcodes(ds: Dataset, *, p: int) -> Dataset:
    y = np.asarray(ds.values, dtype=float)
    if y.shape[0] <= p:
        raise ValueError("not enough observations after trimming")

    y2 = y[p:, :]
    time_index = ds.time_index
    if time_index is not None:
        time_index = time_index[p:]

    if not np.all(np.isfinite(y2)):
        raise ValueError("non-finite values remain after trimming")

    return Dataset.from_arrays(values=y2, variables=list(ds.variables), time_index=time_index)


def _mask_between(targets: list[pd.Period], *, start: pd.Period, end: pd.Period) -> np.ndarray:
    return np.array([start <= t <= end for t in targets], dtype=bool)


def _agg_rmse(err: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(err**2, axis=0))


def _agg_mae(err: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(err), axis=0)


def _agg_mean(x: np.ndarray) -> np.ndarray:
    return np.nanmean(x, axis=0)


def _progress(iterable, *, total: int | None, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    if total is None:
        return iterable

    def gen():
        width = 30
        for i, item in enumerate(iterable, start=1):
            filled = int(width * i / total)
            bar = "=" * filled + "-" * (width - filled)
            print(f"\r{desc}: [{bar}] {i}/{total}", end="", file=sys.stderr)
            yield item
        print("", file=sys.stderr)

    return gen()


def _ratio_style(x: float) -> str:
    if not np.isfinite(x):
        return "dim"
    if x < 0.98:
        return "green"
    if x > 1.02:
        return "red"
    return "yellow"


def _print_rich_ratio_table(
    console: "Console",
    *,
    title: str,
    variables: list[str],
    horizons: list[int],
    ratio_mat: np.ndarray,
) -> None:
    t = Table(title=title, show_lines=False, header_style="bold")
    t.add_column("Variable", style="bold", no_wrap=True)
    for h in horizons:
        t.add_column(f"h={h}", justify="right")

    for i, v in enumerate(variables):
        row: list[Text] = [Text(v)]
        for j in range(len(horizons)):
            x = float(ratio_mat[i, j])
            if np.isfinite(x):
                row.append(Text(f"{x:0.2f}", style=_ratio_style(x)))
            else:
                row.append(Text("nan", style="dim"))
        t.add_row(*row)

    console.print(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--benchmark", type=str, choices=["minnesota", "minnesota_sv"], default="minnesota")
    ap.add_argument("--no-rich", action="store_true")
    ap.add_argument("--fit-draws-bench", type=int, default=None)
    ap.add_argument("--fit-burnin-bench", type=int, default=None)
    ap.add_argument("--fit-draws-elb", type=int, default=1500)
    ap.add_argument("--fit-burnin-elb", type=int, default=500)
    ap.add_argument("--forecast-draws", type=int, default=1000)
    args = ap.parse_args()

    use_rich = (not args.no_rich) and (Console is not None)
    console = Console() if use_rich else None

    t0_all = time.perf_counter()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" if args.data_dir is None else Path(args.data_dir)

    variables = ["CPIAUCSL", "GDP", "UNRATE", "FEDFUNDS"]
    tcodes = [5, 5, 1, 1]
    own_lag_means = [0.0 if t not in (1, 4) else 1.0 for t in tcodes]

    vintages = load_vintages_from_dir(data_dir=data_dir)

    eval_start = _period("2009 Q1")
    eval_end = _period("2019 Q4")
    split1_end = _period("2015 Q4")
    split2_start = _period("2016 Q1")

    eval_quarters = list(pd.period_range(eval_start, eval_end, freq="Q"))

    actuals: dict[pd.Period, np.ndarray] = {}
    if use_rich:
        bench_name = "Minnesota-SV" if args.benchmark == "minnesota_sv" else "Minnesota"
        header = Table.grid(padding=(0, 2))
        header.add_column(justify="left")
        header.add_column(justify="left")
        header.add_row("[bold]Data dir[/bold]", str(data_dir))
        header.add_row("[bold]Vars[/bold]", ", ".join(variables))
        header.add_row("[bold]Tcodes[/bold]", ", ".join(str(t) for t in tcodes))
        header.add_row("[bold]Benchmark[/bold]", bench_name)
        header.add_row("[bold]ELB[/bold]", "FEDFUNDS floor=0.25")
        header.add_row("[bold]p[/bold]", "2")
        header.add_row("[bold]Horizons[/bold]", ", ".join(str(h) for h in [4, 8, 16, 24]))
        header.add_row("[bold]Seed[/bold]", str(args.seed))
        header.add_row(
            "[bold]Fit draws[/bold]",
            f"bench={'auto' if args.fit_draws_bench is None else args.fit_draws_bench}, elb={args.fit_draws_elb}",
        )
        header.add_row(
            "[bold]Burn-in[/bold]",
            f"bench={'auto' if args.fit_burnin_bench is None else args.fit_burnin_bench}, elb={args.fit_burnin_elb}",
        )
        header.add_row("[bold]Forecast draws[/bold]", str(args.forecast_draws))
        console.print(Panel(header, title="Replication run", expand=False))

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_actuals = progress.add_task("Loading actuals", total=len(eval_quarters))
            for q in eval_quarters:
                if q in vintages:
                    ds_q = dataset_from_vintage(vintage_df=vintages[q], variables=variables, vintage=q)
                    y_q = tcode_matrix(ds_q.values, tcodes, var_names=variables)
                    actuals[q] = y_q[-1, :].astype(float)
                progress.advance(task_actuals)
    else:
        for q in _progress(eval_quarters, total=len(eval_quarters), desc="Loading actuals"):
            if q not in vintages:
                continue
            ds_q = dataset_from_vintage(vintage_df=vintages[q], variables=variables, vintage=q)
            y_q = tcode_matrix(ds_q.values, tcodes, var_names=variables)
            actuals[q] = y_q[-1, :].astype(float)

    horizons = [4, 8, 16, 24]
    hmax = int(max(horizons))

    origins = sorted({q - h for q in actuals for h in horizons if (q - h) in vintages})
    if len(origins) == 0:
        raise RuntimeError("no valid origins found for the requested horizons")

    use_sv = args.benchmark == "minnesota_sv"
    vol = VolatilitySpec(enabled=True) if use_sv else None

    model_bench = ModelSpec(p=2, include_intercept=True, volatility=vol)
    model_elb = ModelSpec(
        p=2,
        include_intercept=True,
        elb=ElbSpec(bound=0.25, applies_to=["FEDFUNDS"], enabled=True),
        volatility=vol,
    )

    errors: dict[str, dict[int, list[np.ndarray]]] = {"bench": {}, "elb": {}}
    crps_vals: dict[str, dict[int, list[np.ndarray]]] = {"bench": {}, "elb": {}}
    targets_by_h: dict[int, list[pd.Period]] = {h: [] for h in horizons}

    for h in horizons:
        errors["bench"][h] = []
        errors["elb"][h] = []
        crps_vals["bench"][h] = []
        crps_vals["elb"][h] = []

    t_fit = 0.0
    t_fc = 0.0

    if use_rich:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_origins = progress.add_task("Forecast origins", total=len(origins))
            try:
                for origin in origins:
                    progress.update(task_origins, description=f"Forecast origins (last={origin})")

                    ds_o = dataset_from_vintage(vintage_df=vintages[origin], variables=variables, vintage=origin)
                    y_o = tcode_matrix(ds_o.values, tcodes, var_names=variables)
                    ds_o_t = Dataset.from_arrays(values=y_o, variables=variables, time_index=ds_o.time_index)
                    ds_fit = _trim_after_tcodes(ds_o_t, p=model_bench.p)

                    prior = PriorSpec.niw_minnesota(
                        p=model_bench.p,
                        y=ds_fit.values,
                        include_intercept=model_bench.include_intercept,
                        lambda1=0.05,
                        lambda2=0.5,
                        lambda3=1.0,
                        lambda4=100.0,
                        own_lag_means=own_lag_means,
                    )

                    if args.fit_draws_bench is None:
                        fit_draws_bench = int(args.fit_draws_elb) if use_sv else 1
                    else:
                        fit_draws_bench = int(args.fit_draws_bench)

                    if args.fit_burnin_bench is None:
                        fit_burnin_bench = int(args.fit_burnin_elb) if use_sv else 1
                    else:
                        fit_burnin_bench = int(args.fit_burnin_bench)

                    sampler_bench = SamplerConfig(draws=fit_draws_bench, burn_in=fit_burnin_bench, thin=1)
                    sampler_elb = SamplerConfig(draws=int(args.fit_draws_elb), burn_in=int(args.fit_burnin_elb), thin=1)

                    t0 = time.perf_counter()
                    fit_bench = fit(
                        ds_fit,
                        model_bench,
                        prior,
                        sampler_bench,
                        rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=0, origin=origin, stage=0)),
                    )
                    fit_elb = fit(
                        ds_fit,
                        model_elb,
                        prior,
                        sampler_elb,
                        rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=1, origin=origin, stage=0)),
                    )
                    t_fit += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    fc_bench = forecast(
                        fit_bench,
                        horizons=horizons,
                        draws=int(args.forecast_draws),
                        rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=0, origin=origin, stage=1)),
                    )
                    fc_elb = forecast(
                        fit_elb,
                        horizons=horizons,
                        draws=int(args.forecast_draws),
                        rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=1, origin=origin, stage=1)),
                    )
                    t_fc += time.perf_counter() - t0

                    for h in horizons:
                        target = origin + h
                        if target not in actuals:
                            continue

                        y_true = actuals[target]

                        mean_b = fc_bench.mean[h - 1, :]
                        mean_e = fc_elb.mean[h - 1, :]

                        draws_b = fc_bench.draws[:, h - 1, :]
                        draws_e = fc_elb.draws[:, h - 1, :]

                        targets_by_h[h].append(target)

                        errors["bench"][h].append(y_true - mean_b)
                        errors["elb"][h].append(y_true - mean_e)

                        crps_b = np.array([crps_draws(y_true[j], draws_b[:, j]) for j in range(len(variables))], dtype=float)
                        crps_e = np.array([crps_draws(y_true[j], draws_e[:, j]) for j in range(len(variables))], dtype=float)

                        crps_vals["bench"][h].append(crps_b)
                        crps_vals["elb"][h].append(crps_e)

                    progress.advance(task_origins)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted[/bold yellow] — partial results will be summarized from completed origins.")
    else:
        origin_iter = _progress(origins, total=len(origins), desc="Forecast origins")
        for origin in origin_iter:
            ds_o = dataset_from_vintage(vintage_df=vintages[origin], variables=variables, vintage=origin)
            y_o = tcode_matrix(ds_o.values, tcodes, var_names=variables)
            ds_o_t = Dataset.from_arrays(values=y_o, variables=variables, time_index=ds_o.time_index)
            ds_fit = _trim_after_tcodes(ds_o_t, p=model_bench.p)

            prior = PriorSpec.niw_minnesota(
                p=model_bench.p,
                y=ds_fit.values,
                include_intercept=model_bench.include_intercept,
                lambda1=0.05,
                lambda2=0.5,
                lambda3=1.0,
                lambda4=100.0,
                own_lag_means=own_lag_means,
            )

            if args.fit_draws_bench is None:
                fit_draws_bench = int(args.fit_draws_elb) if use_sv else 1
            else:
                fit_draws_bench = int(args.fit_draws_bench)

            if args.fit_burnin_bench is None:
                fit_burnin_bench = int(args.fit_burnin_elb) if use_sv else 1
            else:
                fit_burnin_bench = int(args.fit_burnin_bench)

            sampler_bench = SamplerConfig(draws=fit_draws_bench, burn_in=fit_burnin_bench, thin=1)
            sampler_elb = SamplerConfig(draws=int(args.fit_draws_elb), burn_in=int(args.fit_burnin_elb), thin=1)

            t0 = time.perf_counter()
            fit_bench = fit(
                ds_fit,
                model_bench,
                prior,
                sampler_bench,
                rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=0, origin=origin, stage=0)),
            )
            fit_elb = fit(
                ds_fit,
                model_elb,
                prior,
                sampler_elb,
                rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=1, origin=origin, stage=0)),
            )
            t_fit += time.perf_counter() - t0

            t0 = time.perf_counter()
            fc_bench = forecast(
                fit_bench,
                horizons=horizons,
                draws=int(args.forecast_draws),
                rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=0, origin=origin, stage=1)),
            )
            fc_elb = forecast(
                fit_elb,
                horizons=horizons,
                draws=int(args.forecast_draws),
                rng=np.random.default_rng(_seed_for(base_seed=args.seed, model_id=1, origin=origin, stage=1)),
            )
            t_fc += time.perf_counter() - t0

            for h in horizons:
                target = origin + h
                if target not in actuals:
                    continue

                y_true = actuals[target]

                mean_b = fc_bench.mean[h - 1, :]
                mean_e = fc_elb.mean[h - 1, :]

                draws_b = fc_bench.draws[:, h - 1, :]
                draws_e = fc_elb.draws[:, h - 1, :]

                targets_by_h[h].append(target)

                errors["bench"][h].append(y_true - mean_b)
                errors["elb"][h].append(y_true - mean_e)

                crps_b = np.array([crps_draws(y_true[j], draws_b[:, j]) for j in range(len(variables))], dtype=float)
                crps_e = np.array([crps_draws(y_true[j], draws_e[:, j]) for j in range(len(variables))], dtype=float)

                crps_vals["bench"][h].append(crps_b)
                crps_vals["elb"][h].append(crps_e)

    def _summarize(metric_name: str, agg_fn, values: dict[str, dict[int, list[np.ndarray]]]) -> None:
        bench_name = "Minnesota-SV" if use_sv else "Minnesota"
        for sample_name, (s0, s1) in {
            "full": (eval_start, eval_end),
            "2009Q1-2015Q4": (eval_start, split1_end),
            "2016Q1-2019Q4": (split2_start, eval_end),
        }.items():
            ratio_mat = np.full((len(variables), len(horizons)), np.nan, dtype=float)
            for hi, h in enumerate(horizons):
                tgt = targets_by_h[h]
                if len(tgt) == 0:
                    continue

                mask = _mask_between(tgt, start=s0, end=s1)
                if not np.any(mask):
                    continue

                b_arr = np.stack(values["bench"][h], axis=0)[mask, :]
                e_arr = np.stack(values["elb"][h], axis=0)[mask, :]

                b = agg_fn(b_arr)
                e = agg_fn(e_arr)
                ratio_mat[:, hi] = e / b

            title = metric_name + f" ratio (ELB / {bench_name}) [{sample_name}]"
            if use_rich:
                _print_rich_ratio_table(console, title=title, variables=variables, horizons=horizons, ratio_mat=ratio_mat)
                console.print("")
            else:
                df = pd.DataFrame(ratio_mat, index=variables, columns=[f"h={h}" for h in horizons])
                df.index.name = title
                print(df.to_string(float_format=lambda x: f"{x:0.2f}"))
                print("")

    _summarize("RMSE", _agg_rmse, errors)
    _summarize("MAE", _agg_mae, errors)
    _summarize("CRPS", _agg_mean, crps_vals)

    if use_rich:
        elapsed = time.perf_counter() - t0_all
        footer = Table.grid(padding=(0, 2))
        footer.add_column(justify="left")
        footer.add_column(justify="left")
        footer.add_row("[bold]Elapsed[/bold]", f"{elapsed:0.1f}s")
        footer.add_row("[bold]Fit time[/bold]", f"{t_fit:0.1f}s")
        footer.add_row("[bold]Forecast time[/bold]", f"{t_fc:0.1f}s")
        console.print(Panel(footer, title="Timing", expand=False))


if __name__ == "__main__":
    main()
