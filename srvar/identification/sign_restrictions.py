from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SignRestriction:
    """Parsed sign restriction for IRFs.

    Indices refer to:
    - `response`: variable index (0..N-1)
    - `shock`: shock index (0..N-1)
    - `horizons`: IRF horizons (0-indexed, matching IRF horizon values)
    """

    response: int
    shock: int
    horizons: np.ndarray
    sign: int  # -1, 0, +1
    cumulative: bool = False


def _parse_sign(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        iv = int(value)
        if iv in (-1, 0, 1):
            return iv
        raise ValueError("sign must be one of -1, 0, +1")

    if not isinstance(value, str) or not value:
        raise ValueError("sign must be a string (+, -, 0) or an integer (-1, 0, +1)")

    v = value.strip().lower()
    if v in {"+", "pos", "positive", "p", "plus", "up"}:
        return 1
    if v in {"-", "neg", "negative", "n", "minus", "down"}:
        return -1
    if v in {"0", "zero"}:
        return 0
    raise ValueError("sign must be one of: +, -, 0")


def _parse_horizon_list(value: Any) -> list[int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        hi = int(value)
        if hi < 0:
            raise ValueError("horizons must be >= 0")
        return [hi]

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("horizons must be an int or a sequence of ints")

    hs: list[int] = []
    for h in value:
        if not isinstance(h, (int, np.integer)) or isinstance(h, bool):
            raise ValueError("horizons must contain only integers")
        hi = int(h)
        if hi < 0:
            raise ValueError("horizons must contain only non-negative integers")
        hs.append(hi)
    if len(hs) < 1:
        raise ValueError("horizons must be non-empty")
    if len(set(hs)) != len(hs):
        raise ValueError("horizons must not contain duplicates")
    return sorted(hs)


def parse_sign_restrictions(
    restrictions: dict[str, Any],
    *,
    variables: Sequence[str],
    n_shocks: int | None = None,
) -> tuple[list[str], list[SignRestriction], bool]:
    """Parse a nested sign-restrictions dict into index-based constraints.

    Parameters
    ----------
    restrictions:
        Mapping: ``shock_name -> variable_name -> spec``.

        Each ``spec`` can be either:
        - Mapping ``horizon -> sign`` (e.g. ``{0: "+", 1: "+"}``)
        - Object spec with keys:
          - ``sign``: "+", "-", "0" (or -1, 0, 1)
          - ``horizons``: int or list[int]
          - ``cumulative``: bool (optional; default False)
    variables:
        Variable names in the IRF output ordering.
    n_shocks:
        Total number of shocks. Defaults to ``len(variables)``.

    Returns
    -------
    (shock_names, parsed_restrictions, has_cumulative)
    """
    if not isinstance(restrictions, dict):
        raise ValueError("restrictions must be a dict: shock -> variable -> restriction spec")

    n = int(len(variables) if n_shocks is None else n_shocks)
    if n < 1:
        raise ValueError("n_shocks must be >= 1")

    var_to_idx = {str(name): i for i, name in enumerate(list(variables))}
    out: list[SignRestriction] = []
    shock_names: list[str] = []
    has_cumulative = False

    if restrictions:
        for shock_idx, (shock_name, shock_spec) in enumerate(restrictions.items()):
            if not isinstance(shock_name, str) or not shock_name:
                raise ValueError("restriction shock names must be non-empty strings")
            if shock_idx >= n:
                raise ValueError(
                    f"restrictions specify {shock_idx + 1} shocks, but n_shocks is {n}"
                )
            shock_names.append(shock_name)

            if not isinstance(shock_spec, dict):
                raise ValueError(f"restrictions for shock {shock_name} must be a dict")

            for var_name, var_spec in shock_spec.items():
                if not isinstance(var_name, str) or not var_name:
                    raise ValueError(
                        f"invalid variable name in restrictions for shock {shock_name}"
                    )
                if var_name not in var_to_idx:
                    raise ValueError(f"restrictions contain unknown variable: {var_name}")
                resp_idx = int(var_to_idx[var_name])

                if not isinstance(var_spec, dict):
                    raise ValueError(
                        f"restrictions for shock {shock_name}, variable {var_name} must be a dict"
                    )

                if any(k in var_spec for k in ("sign", "horizons", "cumulative")):
                    if "sign" not in var_spec or "horizons" not in var_spec:
                        raise ValueError(
                            f"object restriction for shock {shock_name}, variable {var_name} "
                            "must include 'sign' and 'horizons'"
                        )
                    sign = _parse_sign(var_spec["sign"])
                    hs = _parse_horizon_list(var_spec["horizons"])
                    cumulative = bool(var_spec.get("cumulative", False))
                    has_cumulative = has_cumulative or cumulative
                    out.append(
                        SignRestriction(
                            response=resp_idx,
                            shock=shock_idx,
                            horizons=np.asarray(hs, dtype=int),
                            sign=sign,
                            cumulative=cumulative,
                        )
                    )
                else:
                    # horizon -> sign mapping
                    for h_key, sign_val in var_spec.items():
                        if isinstance(h_key, (int, np.integer)) and not isinstance(h_key, bool):
                            hi = int(h_key)
                        elif isinstance(h_key, str) and h_key.strip():
                            try:
                                hi = int(h_key)
                            except ValueError as e:
                                raise ValueError(
                                    f"invalid horizon key in restrictions for shock {shock_name}, "
                                    f"variable {var_name}: {h_key}"
                                ) from e
                        else:
                            raise ValueError(
                                f"invalid horizon key in restrictions for shock {shock_name}, "
                                f"variable {var_name}: {h_key}"
                            )
                        if hi < 0:
                            raise ValueError("restriction horizons must be >= 0")
                        out.append(
                            SignRestriction(
                                response=resp_idx,
                                shock=shock_idx,
                                horizons=np.asarray([hi], dtype=int),
                                sign=_parse_sign(sign_val),
                                cumulative=False,
                            )
                        )

    if len(shock_names) < n:
        used = set(shock_names)
        for j in range(len(shock_names), n):
            base = f"shock{j + 1}"
            name = base
            k = 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            shock_names.append(name)

    return shock_names, out, has_cumulative


def _random_orthonormal(n: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((int(n), int(n)))
    q, r = np.linalg.qr(z)
    diag = np.sign(np.diag(r))
    diag[diag == 0.0] = 1.0
    return q * diag.reshape(1, -1)


def _check_sign_restrictions(
    theta: np.ndarray,
    restrictions: Sequence[SignRestriction],
    *,
    sign_tol: float,
    zero_tol: float,
    theta_cum: np.ndarray | None,
) -> bool:
    for r in restrictions:
        src = theta_cum if r.cumulative else theta
        if src is None:
            raise ValueError(
                "internal error: cumulative restrictions requested but theta_cum is None"
            )
        vals = src[r.horizons, r.response, r.shock]
        if r.sign == 0:
            if np.any(np.abs(vals) > zero_tol):
                return False
        else:
            if np.any(r.sign * vals < sign_tol):
                return False
    return True


def sample_sign_restricted_irf(
    *,
    reduced_irf: np.ndarray,
    impact: np.ndarray,
    restrictions: Sequence[SignRestriction],
    max_attempts: int,
    rng: np.random.Generator,
    sign_tol: float = 0.0,
    zero_tol: float = 0.0,
    has_cumulative: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw a random rotation that satisfies sign restrictions and return structural IRFs.

    Parameters
    ----------
    reduced_irf:
        Reduced-form IRFs with shape ``(H, N, N)`` for horizons ``0..H-1``.
    impact:
        Base impact matrix ``P`` with shape ``(N, N)`` such that ``Sigma = P P'``.
    restrictions:
        Parsed sign restrictions (index-based).
    max_attempts:
        Maximum random rotations to try.
    rng:
        NumPy RNG.
    sign_tol:
        For sign restrictions (+/-), require ``sign * response >= sign_tol``.
    zero_tol:
        For zero restrictions, require ``abs(response) <= zero_tol``.
    has_cumulative:
        Whether any restriction requires cumulative responses (enables fast-path computation).

    Returns
    -------
    (theta, impact_rot, attempts)
        - ``theta`` has shape ``(H, N, N)`` (structural IRFs)
        - ``impact_rot`` is the accepted impact matrix ``P @ Q``
        - ``attempts`` is the number of candidate rotations tried
    """
    if not isinstance(max_attempts, (int, np.integer)) or isinstance(max_attempts, bool):
        raise ValueError("max_attempts must be an integer")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be >= 1")
    max_attempts = int(max_attempts)

    sign_tol_f = float(sign_tol)
    if not np.isfinite(sign_tol_f) or sign_tol_f < 0:
        raise ValueError("sign_tol must be finite and >= 0")

    zero_tol_f = float(zero_tol)
    if not np.isfinite(zero_tol_f) or zero_tol_f < 0:
        raise ValueError("zero_tol must be finite and >= 0")

    red = np.asarray(reduced_irf, dtype=float)
    p = np.asarray(impact, dtype=float)
    if red.ndim != 3:
        raise ValueError("reduced_irf must have shape (H, N, N)")
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("impact must have shape (N, N)")
    if red.shape[1] != p.shape[0] or red.shape[2] != p.shape[0]:
        raise ValueError("reduced_irf and impact shapes are inconsistent")

    theta_cum: np.ndarray | None
    if has_cumulative:
        theta_cum = None  # computed per candidate (depends on rotation)
    else:
        theta_cum = None

    n = int(p.shape[0])
    for attempt in range(1, max_attempts + 1):
        q = _random_orthonormal(n, rng)
        impact_rot = p @ q
        theta = red @ impact_rot

        if not restrictions:
            return theta, impact_rot, attempt

        if has_cumulative:
            theta_cum = np.cumsum(theta, axis=0)

        if _check_sign_restrictions(
            theta,
            restrictions,
            sign_tol=sign_tol_f,
            zero_tol=zero_tol_f,
            theta_cum=theta_cum,
        ):
            return theta, impact_rot, attempt

    raise ValueError(
        "could not find an orthonormal rotation satisfying sign restrictions within "
        f"max_attempts={max_attempts}"
    )
