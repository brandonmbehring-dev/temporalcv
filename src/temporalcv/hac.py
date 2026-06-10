"""
HAC (Heteroskedasticity and Autocorrelation Consistent) Covariance Estimation.

Newey-West and related HAC estimators for valid inference with serially
correlated residuals or scores: kernel weights (Bartlett/Parzen/Quadratic
Spectral), automatic bandwidth selection, a **matrix-accepting** long-run
covariance core (panel-ready), and mean / regression-sandwich standard
errors with optional AR(1) prewhitening.

Ported and modernized from ``dml_ts/dml/hac.py`` (v2.0, issue #9): functions
plus a frozen ``HACResult`` — no stateful estimator class. Value-parity with
the dml_ts implementation is pinned by golden tests for every path that was
correct upstream; deliberate deviations (each found during the port review,
all fail-loud where dml_ts was silent or broken) are listed below.

Deviations from dml_ts (deliberate, documented)
-----------------------------------------------
1. ``HACResult`` carries BOTH ``long_run_variance`` (the long-run variance
   Omega of the series) and ``variance`` (Omega/n — the variance of the MEAN).
   dml_ts exposed only the ambiguous ``get_variance()``, which led its own
   consumer to divide by n twice (dml_ts issue #7: TemporalPLRDML standard
   errors understated by sqrt(n)). **``se`` here is the standard error of the
   mean — never divide it (or ``variance``) by n again.**
2. Prewhitening WORKS here. In dml_ts, ``prewhiten=True`` with a design
   matrix always raised (whitened residuals length n-1 vs n rows of X) and
   mean-mode prewhitening never recolored (wrong scale). Here: whiten
   ``e_t -> u_t = e_t - phi*e_{t-1}``, select the (auto) bandwidth on the
   WHITENED series (Andrews & Monahan 1992), estimate, then recolor by
   ``1/(1-phi)^2`` in both modes; X is row-aligned (first row dropped).
   This is *scalar* AR(1) prewhitening of the residuals with a scalar
   recoloring factor — a simplification of full Andrews-Monahan VAR(1)
   prewhitening of the score series.
3. A singular ``X'X`` raises ``ValueError`` (dml_ts silently fell back to
   the pseudo-inverse, producing garbage covariance for a rank-deficient
   design).
4. A negative leading diagonal in the sandwich covariance (possible with
   the quadratic-spectral kernel in small samples) raises instead of
   silently propagating ``sqrt(negative) = NaN``.
5. A matrix passed where a single series is required raises (dml_ts
   ``ravel()``-ed, silently concatenating columns into one series).

Knowledge Tiers
---------------
[T1] Newey-West long-run variance / sandwich covariance and the Bartlett
     kernel's PSD guarantee: Newey & West (1987).
[T1] Kernel choice and asymptotic optimality of QS: Andrews (1991).
[T1] Automatic bandwidth m = floor(T^(1/3)): Newey & West (1994) rule of
     thumb; Andrews (1991) AR(1) plug-in for the data-driven option.
[T1] AR(1) prewhitening and recoloring: Andrews & Monahan (1992).

References
----------
[T1] Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
     heteroskedasticity and autocorrelation consistent covariance matrix.
     Econometrica, 55(3), 703-708.
[T1] Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
     consistent covariance matrix estimation. Econometrica, 59(3), 817-858.
[T1] Andrews, D. W. K., & Monahan, J. C. (1992). An improved
     heteroskedasticity and autocorrelation consistent covariance matrix
     estimator. Econometrica, 60(4), 953-966.
[T1] Newey, W. K., & West, K. D. (1994). Automatic lag selection in
     covariance matrix estimation. Review of Economic Studies, 61(4), 631-653.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from temporalcv._serialization import result_to_dict
from temporalcv._typing import ArrayLike

__all__ = [
    "HACResult",
    "bartlett_kernel",
    "long_run_covariance",
    "newey_west_covariance",
    "newey_west_se",
    "optimal_bandwidth",
    "parzen_kernel",
    "quadratic_spectral_kernel",
]

KernelName = Literal["bartlett", "parzen", "quadratic_spectral"]
BandwidthMethod = Literal["auto", "newey_west", "andrews"]


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class HACResult:
    """
    Result of a HAC standard-error / covariance estimation.

    Attributes
    ----------
    se : float
        HAC standard error **of the mean / coefficient** — the final,
        directly usable standard error. Do NOT divide by n again.
    variance : float
        ``se**2``: the variance of the mean (mean-mode) or the leading
        diagonal element of the sandwich covariance (regression mode).
    long_run_variance : float | None
        The long-run variance Omega of the (possibly prewhitened-then-
        recolored) series — ``variance * n_samples`` in mean-mode; ``None``
        in regression mode (where the "meat" is a matrix, not a scalar).
    covariance : np.ndarray | None
        Full sandwich covariance matrix ``(k, k)`` in regression mode;
        ``None`` in mean-mode.
    bandwidth : int
        Lag-truncation bandwidth actually used (after auto-selection and
        clamping to ``n - 1``).
    kernel : str
        Kernel name used.
    n_samples : int
        Number of observations in the original (pre-whitening) series.
    effective_dof : float
        ``n_samples - bandwidth`` (heuristic effective degrees of freedom,
        kept for dml_ts parity).
    prewhitened : bool
        Whether AR(1) prewhitening/recoloring was applied.
    ar_coef : float | None
        The fitted AR(1) prewhitening coefficient (``None`` iff
        ``prewhitened`` is False).
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    se: float
    variance: float
    long_run_variance: float | None
    covariance: np.ndarray | None
    bandwidth: int
    kernel: str
    n_samples: int
    effective_dof: float
    prewhitened: bool
    ar_coef: float | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.se) or self.se < 0:
            raise ValueError(f"se must be finite and >= 0, got {self.se}")
        if not np.isfinite(self.variance) or self.variance < 0:
            raise ValueError(f"variance must be finite and >= 0, got {self.variance}")
        if self.bandwidth < 0:
            raise ValueError(f"bandwidth must be >= 0, got {self.bandwidth}")
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {self.n_samples}")
        if self.covariance is not None:
            cov = self.covariance
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError(f"covariance must be square 2-D, got shape {cov.shape}")
        if self.prewhitened != (self.ar_coef is not None):
            raise ValueError(
                f"ar_coef must be set iff prewhitened "
                f"(prewhitened={self.prewhitened}, ar_coef={self.ar_coef})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this HAC result."""
        return result_to_dict(self)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def bartlett_kernel(lag: int, bandwidth: int) -> float:
    """
    Bartlett (triangular) kernel weight: ``1 - |lag|/(bandwidth+1)``.

    The default HAC kernel — simple and guarantees a positive semi-definite
    estimate (Newey & West 1987).

    Returns 1.0 at lag 0 when ``bandwidth <= 0`` (only the variance term
    survives), and 0.0 beyond the bandwidth.
    """
    if bandwidth <= 0:
        return 1.0 if lag == 0 else 0.0
    if abs(lag) <= bandwidth:
        return 1.0 - abs(lag) / (bandwidth + 1)
    return 0.0


def parzen_kernel(lag: int, bandwidth: int) -> float:
    """
    Parzen kernel weight (flatter top than Bartlett; also PSD-guaranteeing).
    """
    if bandwidth <= 0:
        return 1.0 if lag == 0 else 0.0
    x = abs(lag) / (bandwidth + 1)
    if x <= 0.5:
        return 1.0 - 6 * x**2 + 6 * x**3
    elif x <= 1.0:
        return 2.0 * (1 - x) ** 3
    return 0.0


def quadratic_spectral_kernel(lag: int, bandwidth: int) -> float:
    """
    Quadratic spectral (QS) kernel weight.

    Asymptotically MSE-optimal (Andrews 1991) but NOT truncated and can be
    slightly negative at large lags — in small samples the resulting
    estimate may fail positive semi-definiteness (the sandwich path raises
    rather than returning a NaN standard error).
    """
    if lag == 0:
        return 1.0
    if bandwidth <= 0:
        return 0.0
    x = 6 * np.pi * lag / (5 * (bandwidth + 1))
    return float(3 * (np.sin(x) / x - np.cos(x)) / x**2)


_KERNELS: dict[str, Callable[[int, int], float]] = {
    "bartlett": bartlett_kernel,
    "parzen": parzen_kernel,
    "quadratic_spectral": quadratic_spectral_kernel,
}


def _get_kernel(kernel: str) -> Callable[[int, int], float]:
    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Must be one of {sorted(_KERNELS)}")
    return _KERNELS[kernel]


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------


def _as_series(x: ArrayLike, name: str) -> np.ndarray:
    """Normalize to a finite 1-D float series; refuse None/complex/matrix."""
    if x is None:
        raise ValueError(f"{name} is None — expected a 1-D numeric series")
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"{name} has complex dtype {raw.dtype} — refusing to silently discard imaginary parts"
        )
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D series, got shape {arr.shape} "
            f"(a matrix would be silently concatenated by ravel — pass one series)"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _resolve_bandwidth(bandwidth: int | str, series: np.ndarray, kernel: KernelName, n: int) -> int:
    """Resolve an int-or-method bandwidth and clamp to [0, n-1]."""
    if isinstance(bandwidth, str):
        bw = optimal_bandwidth(series, method=bandwidth, kernel=kernel)  # type: ignore[arg-type]
    elif isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)):
        raise TypeError(
            f"bandwidth must be an int or one of 'auto'/'newey_west'/'andrews', "
            f"got {type(bandwidth).__name__} ({bandwidth!r})"
        )
    else:
        bw = int(bandwidth)
        if bw < 0:
            raise ValueError(f"bandwidth must be >= 0, got {bw}")
    return max(0, min(bw, n - 1))


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def optimal_bandwidth(
    residuals: ArrayLike,
    *,
    method: BandwidthMethod = "newey_west",
    kernel: KernelName = "bartlett",
) -> int:
    """
    Automatic lag-truncation bandwidth for HAC estimation.

    Parameters
    ----------
    residuals : ArrayLike
        1-D series (a matrix raises — pass one series).
    method : {"newey_west", "auto", "andrews"}, default "newey_west"
        - ``"newey_west"`` / ``"auto"``: rule of thumb ``floor(T**(1/3))``
          (Newey & West 1994).
        - ``"andrews"``: AR(1) plug-in (Andrews 1991) with kernel-specific
          constants.
    kernel : {"bartlett", "parzen", "quadratic_spectral"}
        Kernel (affects the Andrews constants only).

    Returns
    -------
    int
        Bandwidth >= 0 (0 when fewer than 2 observations).

    Raises
    ------
    ValueError
        On unknown method/kernel or non-1-D input.
    """
    arr = _as_series(residuals, "residuals")
    n = arr.size
    if n < 2:
        return 0

    if method in ("newey_west", "auto"):
        return max(0, int(np.floor(n ** (1 / 3))))

    if method == "andrews":
        _get_kernel(kernel)  # validate kernel name
        if n < 3:
            return 0
        rho = float(np.clip(np.corrcoef(arr[:-1], arr[1:])[0, 1], -0.99, 0.99))
        alpha_den = (1 - rho) ** 4
        alpha = 0.0 if alpha_den < 1e-10 else 4 * rho**2 / alpha_den
        if kernel == "bartlett":
            bw = int(1.1447 * (alpha * n) ** (1 / 3))
        elif kernel == "parzen":
            bw = int(2.6614 * (alpha * n) ** (1 / 5))
        else:  # quadratic_spectral
            bw = int(1.3221 * (alpha * n) ** (1 / 5))
        return max(0, min(bw, n - 1))

    raise ValueError(f"Unknown bandwidth method '{method}'. Use 'newey_west', 'andrews', or 'auto'")


# ---------------------------------------------------------------------------
# Long-run (co)variance — the matrix-accepting core
# ---------------------------------------------------------------------------


def long_run_covariance(
    scores: ArrayLike,
    *,
    bandwidth: int | str = "newey_west",
    kernel: KernelName = "bartlett",
) -> np.ndarray:
    """
    Kernel-weighted long-run covariance of a score series — panel-ready.

    Computes ``Omega = Gamma_0 + sum_j w(j) * (Gamma_j + Gamma_j')`` where
    ``Gamma_j = U[j:]' U[:-j] / n`` are the (biased, 1/n) autocovariance
    matrices of the column-demeaned scores.

    Parameters
    ----------
    scores : ArrayLike
        Score series, shape ``(n,)`` or ``(n, k)``. A 1-D series is treated
        as ``(n, 1)``.
    bandwidth : int | {"auto", "newey_west", "andrews"}, default "newey_west"
        Lag truncation. ``"andrews"`` requires ``k == 1`` (its AR(1) plug-in
        is defined for a single series); for matrices, compute a bandwidth
        explicitly on a chosen series and pass the int.
    kernel : {"bartlett", "parzen", "quadratic_spectral"}
        Kernel weights. Bartlett/Parzen guarantee a PSD estimate; QS may
        not be PSD in small samples (no silent clamping is applied here —
        validate with :func:`temporalcv.psd` if PSD-ness is required).

    Returns
    -------
    np.ndarray
        ``(k, k)`` long-run covariance — always 2-D, including ``(1, 1)``
        for a single series. NOT divided by n: this is Omega, the long-run
        covariance of the score process; the covariance of the score MEAN
        is ``Omega / n``.

    Raises
    ------
    ValueError
        On None/complex/non-finite input, ndim > 2, fewer than 2 rows, or
        ``bandwidth="andrews"`` with ``k > 1``.
    """
    if scores is None:
        raise ValueError("scores is None — expected a numeric series or matrix")
    raw = np.asarray(scores)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"scores has complex dtype {raw.dtype} — refusing to silently discard imaginary parts"
        )
    u = np.asarray(raw, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.ndim != 2:
        raise ValueError(f"scores must be (n,) or (n, k), got shape {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("scores contains non-finite values")
    n, k = u.shape
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")
    if isinstance(bandwidth, str) and bandwidth == "andrews" and k > 1:
        raise ValueError(
            "bandwidth='andrews' is defined for a single series (k=1); "
            "compute a bandwidth on a chosen series and pass the int"
        )

    bw_series = u[:, 0]
    bw = _resolve_bandwidth(bandwidth, bw_series, kernel, n)
    kernel_func = _get_kernel(kernel)

    u = u - u.mean(axis=0)
    omega = (u.T @ u) / n
    for j in range(1, bw + 1):
        if j >= n:
            break
        gamma_j = (u[j:].T @ u[:-j]) / n
        weight = kernel_func(j, bw)
        omega = omega + weight * (gamma_j + gamma_j.T)
    return np.asarray((omega + omega.T) / 2.0, dtype=np.float64)


def _scalar_long_run_variance(
    residuals: np.ndarray, bandwidth: int, kernel_func: Callable[[int, int], float]
) -> float:
    """Scalar long-run variance with the dml_ts non-negativity clamp.

    Identical math to the (1, 1) ``long_run_covariance`` case, plus the
    upstream ``max(0, .)`` clamp that keeps a heavily negative-weighted QS
    estimate from producing a negative variance for the mean (parity with
    dml_ts ``_compute_long_run_variance``).
    """
    n = residuals.size
    e = residuals - residuals.mean()
    omega = float(np.sum(e**2) / n)
    for j in range(1, bandwidth + 1):
        if j >= n:
            break
        gamma_j = float(np.sum(e[j:] * e[:-j]) / n)
        omega += 2 * kernel_func(j, bandwidth) * gamma_j
    return max(0.0, omega)


# ---------------------------------------------------------------------------
# AR(1) prewhitening (Andrews & Monahan 1992, scalar simplification)
# ---------------------------------------------------------------------------


def _prewhiten(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ``e_t = phi*e_{t-1} + u_t`` by OLS; return (u, clipped phi)."""
    if residuals.size < 3:
        raise ValueError(f"prewhitening requires at least 3 observations, got {residuals.size}")
    e_lag, e_curr = residuals[:-1], residuals[1:]
    denom = float(np.sum(e_lag**2))
    if denom == 0.0:
        raise ValueError("prewhitening is undefined for a constant zero series")
    phi = float(np.clip(np.sum(e_lag * e_curr) / denom, -0.99, 0.99))
    return e_curr - phi * e_lag, phi


# ---------------------------------------------------------------------------
# Newey-West standard errors / sandwich covariance
# ---------------------------------------------------------------------------


def newey_west_se(
    residuals: ArrayLike,
    X: ArrayLike | None = None,
    *,
    bandwidth: int | str = "auto",
    kernel: KernelName = "bartlett",
    prewhiten: bool = False,
) -> HACResult:
    """
    Newey-West HAC standard error of a mean (or regression coefficient).

    With ``X=None`` (mean-mode): the HAC standard error of the sample mean
    of ``residuals`` — e.g. the SE of an estimator with influence
    representation ``theta_hat - theta ~= mean(psi_i)`` when ``residuals``
    are the influence scores. With ``X``: the leading-coefficient SE from
    the sandwich covariance (see :func:`newey_west_covariance`).

    **The returned ``se`` is final — do not divide by n again** (dml_ts
    issue #7 was exactly that mistake against an ambiguous variance
    accessor).

    Parameters
    ----------
    residuals : ArrayLike
        1-D series (residuals or influence scores), n >= 2.
    X : ArrayLike, optional
        Design matrix ``(n, k)``; switches to the sandwich estimator.
    bandwidth : int | {"auto", "newey_west", "andrews"}, default "auto"
        Lag truncation; strings auto-select (``"auto"`` == ``"newey_west"``,
        the ``floor(T**(1/3))`` rule). With ``prewhiten=True``, auto
        selection runs on the WHITENED series (Andrews & Monahan 1992).
    kernel : {"bartlett", "parzen", "quadratic_spectral"}
        Kernel weights.
    prewhiten : bool, default False
        Apply AR(1) prewhitening and ``1/(1-phi)^2`` recoloring (requires
        n >= 3).

    Returns
    -------
    HACResult
        With ``se``, ``variance`` (= se**2), ``long_run_variance`` (Omega;
        mean-mode only), ``covariance`` (sandwich mode only), and the
        bandwidth/kernel/prewhitening metadata.

    Raises
    ------
    ValueError
        On invalid inputs, n < 2, or prewhitening with n < 3.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> e = rng.standard_normal(200)
    >>> result = newey_west_se(e, bandwidth=4)
    >>> bool(result.se > 0)
    True
    >>> round(result.variance * result.n_samples, 12) == round(
    ...     result.long_run_variance, 12
    ... )
    True
    """
    e = _as_series(residuals, "residuals")
    n = e.size
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    if X is not None:
        return newey_west_covariance(e, X, bandwidth=bandwidth, kernel=kernel, prewhiten=prewhiten)

    kernel_func = _get_kernel(kernel)
    ar_coef: float | None = None
    if prewhiten:
        work, ar_coef = _prewhiten(e)
    else:
        work = e
    bw = _resolve_bandwidth(bandwidth, work, kernel, work.size)

    omega = _scalar_long_run_variance(work, bw, kernel_func)
    if ar_coef is not None:
        omega = omega / (1.0 - ar_coef) ** 2

    variance = omega / n
    return HACResult(
        se=float(np.sqrt(variance)),
        variance=float(variance),
        long_run_variance=float(omega),
        covariance=None,
        bandwidth=bw,
        kernel=kernel,
        n_samples=n,
        effective_dof=float(n - bw),
        prewhitened=prewhiten,
        ar_coef=ar_coef,
    )


def newey_west_covariance(
    residuals: ArrayLike,
    X: ArrayLike,
    *,
    bandwidth: int | str = "auto",
    kernel: KernelName = "bartlett",
    prewhiten: bool = False,
) -> HACResult:
    """
    Newey-West sandwich covariance for regression coefficients.

    ``V = (X'X)^-1 * Omega * (X'X)^-1`` with the HAC "meat"
    ``Omega = sum_j w(j) sum_i (e_i x_i)(e_{i+j} x_{i+j})'`` built from the
    score series ``u_i = e_i * x_i``.

    Parameters
    ----------
    residuals : ArrayLike
        Regression residuals, 1-D, length n.
    X : ArrayLike
        Design matrix ``(n, k)`` (a 1-D X is treated as one column).
    bandwidth, kernel, prewhiten
        As in :func:`newey_west_se`. Prewhitening whitens the RESIDUALS
        (scalar AR(1)), drops the first row of X to stay aligned, and
        recolors the covariance by ``1/(1-phi)^2`` — a scalar
        simplification of Andrews-Monahan VAR(1) score prewhitening.

    Returns
    -------
    HACResult
        ``covariance`` holds the ``(k, k)`` sandwich; ``se``/``variance``
        are the leading coefficient's (index 0). ``long_run_variance`` is
        None in this mode.

    Raises
    ------
    ValueError
        On invalid inputs, length mismatch, n < max(2, k), singular ``X'X``
        (rank-deficient design — dml_ts silently used a pseudo-inverse
        here), or a negative leading variance (possible with the QS kernel
        in small samples — dml_ts silently returned NaN).
    """
    e = _as_series(residuals, "residuals")
    if X is None:
        raise ValueError("X is None — use newey_west_se(residuals) for the mean-mode SE")
    raw_x = np.asarray(X)
    if np.iscomplexobj(raw_x):
        raise ValueError(
            f"X has complex dtype {raw_x.dtype} — refusing to silently discard imaginary parts"
        )
    x = np.asarray(raw_x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"X must be (n, k), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains non-finite values")

    n_original = e.size
    if e.size != x.shape[0]:
        raise ValueError(f"residuals length ({e.size}) != X rows ({x.shape[0]})")

    ar_coef: float | None = None
    if prewhiten:
        e, ar_coef = _prewhiten(e)
        x = x[1:]  # row-align with the whitened residuals (dml_ts crashed here)

    n, k = x.shape
    if n < k:
        raise ValueError(f"Need n >= k, got n={n}, k={k}")
    if n < 2:
        raise ValueError(f"Need at least 2 observations, got {n}")

    bw = _resolve_bandwidth(bandwidth, e, kernel, n)
    kernel_func = _get_kernel(kernel)

    xtx = x.T @ x
    # np.linalg.inv does not reliably raise on float-singular matrices
    # (duplicate columns can "invert" into huge garbage values), so check
    # rank explicitly. dml_ts silently used a pseudo-inverse here.
    if np.linalg.matrix_rank(xtx) < k:
        raise ValueError(
            "X'X is singular — the design matrix is rank-deficient; drop collinear columns"
        )
    xtx_inv = np.linalg.inv(xtx)

    u = e.reshape(-1, 1) * x
    omega = u.T @ u
    for j in range(1, bw + 1):
        if j >= n:
            break
        weight = kernel_func(j, bw)
        gamma_j = u[j:].T @ u[:-j]
        omega += weight * (gamma_j + gamma_j.T)

    cov = xtx_inv @ omega @ xtx_inv
    cov = np.asarray((cov + cov.T) / 2.0, dtype=np.float64)
    if ar_coef is not None:
        cov = cov / (1.0 - ar_coef) ** 2

    leading = float(cov[0, 0])
    if leading < 0:
        raise ValueError(
            f"leading sandwich variance is negative ({leading:.3e}) — the "
            f"'{kernel}' kernel can fail PSD-ness in small samples; use "
            f"'bartlett' or 'parzen', or a smaller bandwidth"
        )

    return HACResult(
        se=float(np.sqrt(leading)),
        variance=leading,
        long_run_variance=None,
        covariance=cov,
        bandwidth=bw,
        kernel=kernel,
        n_samples=n_original,
        effective_dof=float(n_original - bw),
        prewhitened=prewhiten,
        ar_coef=ar_coef,
    )
