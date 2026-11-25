"""TBA-trained error-correction model for FRC vision scout.

Learns a per-component affine calibration  ``tba_actual = a * vision_est + b``
from paired (vision_estimate, tba_actual) observations collected across matches.
Corrects systematic vision bias (e.g. persistent fuel undercount) and attaches
empirical confidence intervals based on training-set residual scatter.

Usage
-----
    from calib.error_model import ErrorCorrectionModel, CalibrationArtifact

    # --- offline training (run once per event, post-TBA) ---
    model = ErrorCorrectionModel()
    artifact = model.fit(pairs_jsonl_path="calib/data/all_pairs.jsonl")
    artifact.save("calib/artifacts/2026miket-v1.json")

    # --- live application ---
    artifact = CalibrationArtifact.load("calib/artifacts/2026miket-v1.json")
    result = artifact.apply({"total_score": 142.0, "fuel_total": 87.0})
    # result["total_score_cal"], result["total_score_cal_ci_lo"], ...

Public API
----------
    PairedSample          dataclass — one (match, alliance, component) observation
    ComponentCalib        dataclass — fitted (a, b, sigma) for one component
    CalibrationArtifact   serializable container of per-component calibs + metadata
    ErrorCorrectionModel  fit() + evaluate() entry point
    load_pairs_jsonl()    reads calib/data/*.jsonl into List[PairedSample]
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum total matched pairs (across ALL components) required to attempt any fit.
MIN_MATCHES_TOTAL: int = 8

#: Minimum pairs for a single component to receive its own affine fit.
#: Below this, that component gets the identity calibration (a=1, b=0).
MIN_MATCHES_PER_COMPONENT: int = 6

#: Minimum holdout pairs to report a meaningful holdout evaluation.
MIN_HOLDOUT: int = 3

#: Holdout fraction (last fraction of matches by sorted key).
HOLDOUT_FRAC: float = 0.20

#: Components the model supports. Only these are calibrated; others pass through raw.
#: Per SCHEMA_VERIFIED.md the ONLY valid vision→TBA calibration targets are the FUEL
#: counts (vision emits fuel only). tower_points/foul_points were removed: vision never
#: produces them, so they could only ever be identity and their presence was misleading.
#: total_score is retained ONLY for synthetic unit tests; the real collector never emits
#: total_score pairs (it forbids vision-total→TBA-total), so it stays identity in practice.
SUPPORTED_COMPONENTS: Tuple[str, ...] = (
    "fuel_total",
    "fuel_auto",
    "fuel_teleop",
    "total_score",
)

#: Calibration is considered useful only if holdout MAE reduction exceeds this percent.
MIN_USEFUL_MAE_REDUCTION_PCT: float = 5.0

#: z-score for the CI (1.96 ≈ 95% for Gaussian residuals).
CI_Z: float = 1.96


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PairedSample:
    """One (match, alliance, component) observation pairing vision to TBA truth.

    Fields mirror the JSONL schema defined in DESIGN_error_model.md §1.1.
    """
    match_key: str
    comp_level: str           # "qm" | "sf" | "f"
    alliance: str             # "blue" | "red"
    component: str            # must be in SUPPORTED_COMPONENTS
    vision_est: float         # raw pipeline output
    tba_actual: float         # TBA ground truth
    event_key: str = ""
    timestamp_utc: Optional[str] = None

    @property
    def residual(self) -> float:
        """tba_actual - vision_est  (positive = vision underestimates)."""
        return self.tba_actual - self.vision_est


@dataclass
class ComponentCalib:
    """Fitted affine calibration for one score component.

    Calibrated estimate = ``a * vision_est + b``.
    ``sigma`` is the std-dev of training residuals (tba - calibrated) and drives the CI.
    """
    component: str
    a: float                  # scale factor; 1.0 = no rescaling needed
    b: float                  # offset; 0.0 = no bias
    sigma: float              # residual std-dev on training set
    n_train: int              # number of training pairs used
    is_identity: bool = False # True when fewer than MIN_MATCHES_PER_COMPONENT pairs

    def apply(self, vision_est: float) -> Tuple[float, float, float]:
        """Return (calibrated, ci_lo, ci_hi).

        Parameters
        ----------
        vision_est : float
            Raw vision pipeline estimate for this component.

        Returns
        -------
        calibrated : float
            Affine-corrected estimate.
        ci_lo, ci_hi : float
            Empirical 95% confidence interval based on training residual scatter.
            Interpret as: "given past pipeline behaviour, the true value likely lies here."
        """
        calibrated = self.a * vision_est + self.b
        half_width = CI_Z * self.sigma
        return calibrated, calibrated - half_width, calibrated + half_width


@dataclass
class EvalReport:
    """Holdout-set evaluation metrics for one component."""
    component: str
    n_holdout: int
    mae_raw: Optional[float]
    mae_cal: Optional[float]
    rmse_raw: Optional[float]
    rmse_cal: Optional[float]
    bias_raw: Optional[float]
    bias_cal: Optional[float]
    mae_reduction_pct: Optional[float]
    is_useful: bool           # True if mae_reduction_pct > MIN_USEFUL_MAE_REDUCTION_PCT
    skipped: bool = False     # True when n_holdout < MIN_HOLDOUT
    skip_reason: str = ""


@dataclass
class CalibrationArtifact:
    """Serializable container of all per-component calibrations and training metadata.

    Save/load via JSON to ``calib/artifacts/``.
    """
    event_key: str
    version: str
    n_total_pairs: int
    components: Dict[str, ComponentCalib] = field(default_factory=dict)
    eval_reports: Dict[str, EvalReport] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write artifact to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_key": self.event_key,
            "version": self.version,
            "n_total_pairs": self.n_total_pairs,
            "components": {k: asdict(v) for k, v in self.components.items()},
            "eval_reports": {k: asdict(v) for k, v in self.eval_reports.items()},
            "warnings": self.warnings,
        }
        p.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        """Load artifact from a JSON file produced by ``save()``."""
        data = json.loads(Path(path).read_text())
        components = {
            k: ComponentCalib(**v) for k, v in data.get("components", {}).items()
        }
        eval_reports = {
            k: EvalReport(**v) for k, v in data.get("eval_reports", {}).items()
        }
        return cls(
            event_key=data["event_key"],
            version=data["version"],
            n_total_pairs=data["n_total_pairs"],
            components=components,
            eval_reports=eval_reports,
            warnings=data.get("warnings", []),
        )

    # ------------------------------------------------------------------
    # Live application
    # ------------------------------------------------------------------

    def apply(
        self,
        vision_estimates: Dict[str, float],
        *,
        keep_raw: bool = True,
    ) -> Dict[str, object]:
        """Apply calibration to a dict of vision estimates and return enriched output.

        Parameters
        ----------
        vision_estimates : dict
            Keys are component names (e.g. ``"total_score"``), values are raw floats.
        keep_raw : bool
            If True (default), include ``{component}_raw`` in output alongside
            ``{component}_cal``, ``{component}_cal_ci_lo``, ``{component}_cal_ci_hi``.

        Returns
        -------
        dict
            For each component present in both ``vision_estimates`` and ``self.components``:
                ``{c}_cal``       — calibrated point estimate
                ``{c}_cal_ci_lo`` — lower bound of empirical 95% CI
                ``{c}_cal_ci_hi`` — upper bound
                ``{c}_raw``       — original vision estimate (if keep_raw)
                ``{c}_identity``  — True if calibration was identity (< min data)
            Components not in self.components pass through unchanged with _cal == _raw.
            Metadata: ``cal_version``, ``cal_event_key``.
        """
        out: Dict[str, object] = {
            "cal_version": self.version,
            "cal_event_key": self.event_key,
        }

        for component, raw_val in vision_estimates.items():
            if keep_raw:
                out[f"{component}_raw"] = raw_val

            calib = self.components.get(component)
            if calib is None:
                # No calibration available — pass through as-is
                out[f"{component}_cal"] = raw_val
                out[f"{component}_cal_ci_lo"] = None
                out[f"{component}_cal_ci_hi"] = None
                out[f"{component}_identity"] = True
            else:
                cal, ci_lo, ci_hi = calib.apply(raw_val)
                out[f"{component}_cal"] = cal
                out[f"{component}_cal_ci_lo"] = ci_lo
                out[f"{component}_cal_ci_hi"] = ci_hi
                out[f"{component}_identity"] = calib.is_identity

        return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pairs_jsonl(path: str | Path) -> List[PairedSample]:
    """Load paired observations from a JSONL file.

    Each line must be a JSON object matching the PairedSample schema.
    Lines with missing required fields or unsupported components are skipped
    with a warning — never raise on individual malformed rows.

    Parameters
    ----------
    path : str or Path
        Path to a ``.jsonl`` file (one JSON object per line).

    Returns
    -------
    List[PairedSample]
        All valid records, in file order.
    """
    records: List[PairedSample] = []
    path = Path(path)
    if not path.exists():
        warnings.warn(f"pairs file not found: {path}", stacklevel=2)
        return records

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            warnings.warn(f"line {lineno}: JSON parse error — {exc}", stacklevel=2)
            continue

        # Required fields
        required = ("match_key", "comp_level", "alliance", "component",
                    "vision_est", "tba_actual")
        if any(obj.get(k) is None for k in required):
            warnings.warn(
                f"line {lineno}: missing required field(s) in {obj}", stacklevel=2
            )
            continue

        # Filter to supported components only
        if obj["component"] not in SUPPORTED_COMPONENTS:
            continue

        try:
            records.append(PairedSample(
                match_key=str(obj["match_key"]),
                comp_level=str(obj["comp_level"]),
                alliance=str(obj["alliance"]),
                component=str(obj["component"]),
                vision_est=float(obj["vision_est"]),
                tba_actual=float(obj["tba_actual"]),
                event_key=str(obj.get("event_key", "")),
                timestamp_utc=obj.get("timestamp_utc"),
            ))
        except (TypeError, ValueError) as exc:
            warnings.warn(f"line {lineno}: type conversion error — {exc}", stacklevel=2)

    return records


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class ErrorCorrectionModel:
    """Fit and evaluate per-component affine error-correction calibrations.

    Parameters
    ----------
    mode : str
        ``"ols"`` — ordinary least squares (default, always available).
        ``"huber"`` — Huber regression for outlier robustness (requires sklearn).
    event_key : str
        Identifier embedded in saved artifacts (e.g. ``"2026miket"``).
    version : str
        Artifact version string (e.g. ``"v1"``).

    Examples
    --------
        model = ErrorCorrectionModel(mode="huber", event_key="2026miket")
        artifact = model.fit("calib/data/all_pairs.jsonl")
        print(artifact.eval_reports["total_score"])
        artifact.save("calib/artifacts/2026miket-v1.json")
    """

    def __init__(
        self,
        mode: str = "ols",
        event_key: str = "unknown",
        version: str = "v1",
    ) -> None:
        if mode not in ("ols", "huber"):
            raise ValueError(f"mode must be 'ols' or 'huber', got {mode!r}")
        self.mode = mode
        self.event_key = event_key
        self.version = version

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fit(self, pairs_jsonl_path: str | Path) -> CalibrationArtifact:
        """Fit per-component affine calibrations from paired observations.

        Performs a train/holdout split by match key (last HOLDOUT_FRAC matches as
        holdout), fits the calibration on the training set, and evaluates on holdout.

        Parameters
        ----------
        pairs_jsonl_path : str or Path
            Path to the merged JSONL pairs file (see ``load_pairs_jsonl``).

        Returns
        -------
        CalibrationArtifact
            Fitted calibrations with embedded evaluation reports and any warnings.
            If fewer than MIN_MATCHES_TOTAL total pairs are found, returns an
            artifact containing identity calibrations and a warning.
        """
        samples = load_pairs_jsonl(pairs_jsonl_path)
        return self.fit_from_samples(samples)

    def _fit_core(self, samples: List[PairedSample]) -> CalibrationArtifact:
        artifact = CalibrationArtifact(
            event_key=self.event_key,
            version=self.version,
            n_total_pairs=len(samples),
        )

        if len(samples) < MIN_MATCHES_TOTAL:
            artifact.warnings.append(
                f"Only {len(samples)} pairs available (need {MIN_MATCHES_TOTAL}). "
                "Returning identity calibrations for all components."
            )
            for component in SUPPORTED_COMPONENTS:
                artifact.components[component] = _identity_calib(component)
            return artifact

        # Split by match key
        train_samples, holdout_samples = _train_holdout_split(samples, HOLDOUT_FRAC)

        # Fit each component
        for component in SUPPORTED_COMPONENTS:
            train_c = [s for s in train_samples if s.component == component]
            hold_c = [s for s in holdout_samples if s.component == component]

            if len(train_c) < MIN_MATCHES_PER_COMPONENT:
                calib = _identity_calib(component)
                artifact.warnings.append(
                    f"Component '{component}': only {len(train_c)} train pairs "
                    f"(need {MIN_MATCHES_PER_COMPONENT}). Using identity calibration."
                )
            else:
                calib = self._fit_component(component, train_c)

                # Sanity: clamp negative scale (would invert ordering)
                if calib.a < 0:
                    artifact.warnings.append(
                        f"Component '{component}': fitted a={calib.a:.4f} < 0 — "
                        "clamped to 0.1.  Check for data quality issues."
                    )
                    calib.a = 0.1

            artifact.components[component] = calib
            artifact.eval_reports[component] = _evaluate_holdout(
                component, calib, hold_c
            )

        return artifact

    def fit_from_samples(self, samples: List[PairedSample]) -> CalibrationArtifact:
        """Fit from an already-loaded list of PairedSample objects.

        Convenience wrapper for callers that assemble samples programmatically
        (e.g. in tests or when pairing inline from a TBA API response).

        Parameters
        ----------
        samples : List[PairedSample]
            All samples, any mix of components and alliances.

        Returns
        -------
        CalibrationArtifact
            Same semantics as ``fit()``.
        """
        return self._fit_core(samples)

    # ------------------------------------------------------------------
    # Internal fitting helpers
    # ------------------------------------------------------------------

    def _fit_component(
        self, component: str, train_samples: List[PairedSample]
    ) -> ComponentCalib:
        """Fit affine (a, b) for one component on training samples.

        Uses OLS via numpy when ``self.mode == "ols"``, or HuberRegressor from
        sklearn when ``self.mode == "huber"``.

        Parameters
        ----------
        component : str
        train_samples : List[PairedSample]
            Training observations for this component only.

        Returns
        -------
        ComponentCalib
            Fitted calibration with residual sigma on training set.
        """
        if self.mode == "ols":
            a, b = _fit_ols(train_samples)
        else:  # "huber"
            a, b = _fit_huber(train_samples)

        sigma = _residual_sigma(train_samples, a, b)

        return ComponentCalib(
            component=component,
            a=a,
            b=b,
            sigma=sigma,
            n_train=len(train_samples),
            is_identity=False,
        )


# ---------------------------------------------------------------------------
# Fitting implementations (module-level for testability)
# ---------------------------------------------------------------------------

def _fit_ols(samples: List[PairedSample]) -> Tuple[float, float]:
    """Ordinary least squares fit of  y = a*x + b.

    Solves via numpy.linalg.lstsq on the design matrix [x, 1].

    Parameters
    ----------
    samples : List[PairedSample]

    Returns
    -------
    (a, b) : Tuple[float, float]
    """
    import numpy as np
    if len(samples) < 2:
        return 1.0, 0.0
    x = np.array([s.vision_est for s in samples], dtype=float)
    y = np.array([s.tba_actual for s in samples], dtype=float)
    # Degenerate x (all identical vision estimates): can't fit a slope -> bias-only.
    if float(np.ptp(x)) < 1e-9:
        return 1.0, float(np.mean(y - x))
    coeffs, *_ = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _fit_huber(samples: List[PairedSample]) -> Tuple[float, float]:
    """Huber-robust regression fit of  y = a*x + b.

    Uses sklearn.linear_model.HuberRegressor with epsilon=1.35 (95%-efficiency
    under Gaussian noise while down-weighting gross outliers).

    Parameters
    ----------
    samples : List[PairedSample]

    Returns
    -------
    (a, b) : Tuple[float, float]
    """
    # TODO: implement using sklearn
    # from sklearn.linear_model import HuberRegressor
    # import numpy as np
    # X = np.array([[s.vision_est] for s in samples])
    # y = np.array([s.tba_actual for s in samples])
    # reg = HuberRegressor(epsilon=1.35, fit_intercept=True)
    # reg.fit(X, y)
    # return float(reg.coef_[0]), float(reg.intercept_)
    raise NotImplementedError("_fit_huber: sklearn implementation pending")


def _residual_sigma(
    samples: List[PairedSample], a: float, b: float
) -> float:
    """Compute std-dev of (tba_actual - calibrated) on a sample set.

    Used for CI construction.  Returns 0.0 when len(samples) < 2.

    Parameters
    ----------
    samples : List[PairedSample]
    a, b : float
        Affine coefficients.

    Returns
    -------
    float
    """
    if len(samples) < 2:
        return 0.0
    residuals = [s.tba_actual - (a * s.vision_est + b) for s in samples]
    mean_r = sum(residuals) / len(residuals)
    variance = sum((r - mean_r) ** 2 for r in residuals) / (len(residuals) - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _train_holdout_split(
    samples: List[PairedSample], holdout_frac: float
) -> Tuple[List[PairedSample], List[PairedSample]]:
    """Split samples into train and holdout by match key.

    The last ``ceil(holdout_frac * n_unique_matches)`` match keys (sorted
    lexicographically, approximating chronological order within an event) become
    the holdout.  Both alliances of a held-out match stay in holdout — no leakage.

    Parameters
    ----------
    samples : List[PairedSample]
    holdout_frac : float
        Fraction of unique match keys to hold out (e.g. 0.20).

    Returns
    -------
    (train_samples, holdout_samples)
    """
    unique_keys = sorted(set(s.match_key for s in samples))
    n_holdout = max(1, math.ceil(holdout_frac * len(unique_keys)))
    holdout_keys = set(unique_keys[-n_holdout:])
    train = [s for s in samples if s.match_key not in holdout_keys]
    holdout = [s for s in samples if s.match_key in holdout_keys]
    return train, holdout


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_holdout(
    component: str,
    calib: ComponentCalib,
    holdout: List[PairedSample],
) -> EvalReport:
    """Compute before/after metrics on the holdout set for one component.

    Parameters
    ----------
    component : str
    calib : ComponentCalib
        The fitted calibration to evaluate.
    holdout : List[PairedSample]
        Holdout observations for this component.

    Returns
    -------
    EvalReport
    """
    if len(holdout) < MIN_HOLDOUT:
        return EvalReport(
            component=component,
            n_holdout=len(holdout),
            mae_raw=None, mae_cal=None,
            rmse_raw=None, rmse_cal=None,
            bias_raw=None, bias_cal=None,
            mae_reduction_pct=None,
            is_useful=False,
            skipped=True,
            skip_reason=f"Only {len(holdout)} holdout pairs (need {MIN_HOLDOUT})",
        )

    raw_errs = [abs(s.vision_est - s.tba_actual) for s in holdout]
    cal_errs = [abs(calib.apply(s.vision_est)[0] - s.tba_actual) for s in holdout]

    raw_sq = [e ** 2 for e in raw_errs]
    cal_sq = [e ** 2 for e in cal_errs]

    raw_biases = [s.vision_est - s.tba_actual for s in holdout]
    cal_biases = [calib.apply(s.vision_est)[0] - s.tba_actual for s in holdout]

    mae_raw = sum(raw_errs) / len(raw_errs)
    mae_cal = sum(cal_errs) / len(cal_errs)
    rmse_raw = math.sqrt(sum(raw_sq) / len(raw_sq))
    rmse_cal = math.sqrt(sum(cal_sq) / len(cal_sq))
    bias_raw = sum(raw_biases) / len(raw_biases)
    bias_cal = sum(cal_biases) / len(cal_biases)

    mae_reduction_pct: Optional[float] = None
    if mae_raw and mae_raw > 0:
        mae_reduction_pct = (mae_raw - mae_cal) / mae_raw * 100.0

    is_useful = (
        mae_reduction_pct is not None
        and mae_reduction_pct > MIN_USEFUL_MAE_REDUCTION_PCT
    )

    return EvalReport(
        component=component,
        n_holdout=len(holdout),
        mae_raw=mae_raw,
        mae_cal=mae_cal,
        rmse_raw=rmse_raw,
        rmse_cal=rmse_cal,
        bias_raw=bias_raw,
        bias_cal=bias_cal,
        mae_reduction_pct=mae_reduction_pct,
        is_useful=is_useful,
    )


# ---------------------------------------------------------------------------
# Identity calibration
# ---------------------------------------------------------------------------

def _identity_calib(component: str) -> ComponentCalib:
    """Return a pass-through (a=1, b=0) calibration used when data is insufficient.

    Parameters
    ----------
    component : str

    Returns
    -------
    ComponentCalib
        a=1.0, b=0.0, sigma=0.0, n_train=0, is_identity=True.
    """
    return ComponentCalib(
        component=component,
        a=1.0,
        b=0.0,
        sigma=0.0,
        n_train=0,
        is_identity=True,
    )
