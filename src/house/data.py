"""
Data loading and validation for the two-zone grey-box house model.

Deliberately independent of the RC / Kalman machinery: the daily energy-balance
diagnostic needs a trustworthy dataframe and nothing else.

CSV columns:
    timestamp, T_o, v, GHI, T_i1, T_i2,
    Q_dist1_only, Q_dist2_only, Q_dist_together
    (+ optional T_i1_set, T_i2_set, T_w)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DT = 300                       # timestep [s]
MIN_SEGMENT = 288              # 24 h at 5-min resolution
GHI_MAX = 1100                 # W/m2, plausible clear-sky peak at this latitude

# Weather gaps up to this many steps are interpolated; longer ones are left as
# NaN and break the record into separate segments. One hour is short enough
# that linear interpolation of T_o is honest, and long enough to absorb the
# scattered single-sample dropouts. Interpolating across a whole missing day
# would invent 24 h of weather and feed it to the likelihood as if measured.
MAX_INTERP_STEPS = 12

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "data.csv"

REQUIRED = [
    "timestamp", "T_o", "v", "GHI", "T_i1", "T_i2",
    "Q_dist1_only", "Q_dist2_only", "Q_dist_together",
]
OPTIONAL = ["T_i1_set", "T_i2_set", "T_w", "T_o_F"]
Q_COLS = ["Q_dist1_only", "Q_dist2_only", "Q_dist_together"]
WEATHER = ["T_o", "v", "GHI"]


@dataclass
class Dataset:
    df: pd.DataFrame
    segments: list = field(default_factory=list)   # [(start, stop), ...] row slices
    has_tw: bool = False
    has_setpoints: bool = False

    @property
    def q_total(self) -> pd.Series:
        return self.df[Q_COLS].sum(axis=1)


def load_data(path=DEFAULT_CSV, verbose: bool = True) -> Dataset:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric = [c for c in REQUIRED[1:] + OPTIONAL if c in df.columns]
    for c in numeric:
        before = df[c].isna()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_bad = int((~before & df[c].isna()).sum())
        if n_bad and verbose:
            print(f"  ! {c}: {n_bad} non-numeric value(s) -> NaN")

    dup = int(df["timestamp"].duplicated().sum())
    if dup:
        if verbose:
            print(f"  ! {dup} duplicate timestamps dropped")
        df = df.drop_duplicates("timestamp").reset_index(drop=True)

    has_tw = "T_w" in df.columns
    has_setpoints = "T_i1_set" in df.columns and "T_i2_set" in df.columns

    # Weather gaps: interpolate short ones, leave long ones missing so they
    # break segments rather than being papered over with invented values.
    for c in WEATHER:
        df[c], missing, n_short, n_long, n_runs_long = _fill_short_gaps(
            df[c], MAX_INTERP_STEPS)
        df[f"{c}_missing"] = missing
        if verbose and missing.any():
            print(f"  ! {c}: {int(missing.sum())} missing "
                  f"-> {n_short} interpolated (gaps <= {MAX_INTERP_STEPS} steps), "
                  f"{n_long} left NaN in {n_runs_long} long gap(s)")

    n_clip = int((df["GHI"] > GHI_MAX).sum())
    if n_clip:
        if verbose:
            print(f"  ! GHI: {n_clip} values clipped to {GHI_MAX} W/m2")
        df["GHI"] = df["GHI"].clip(upper=GHI_MAX)

    df["Q_total"] = df[Q_COLS].sum(axis=1)
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

    data = Dataset(
        df=df,
        segments=_segment(df),
        has_tw=has_tw,
        has_setpoints=has_setpoints,
    )
    if verbose:
        summarise(data)
    return data


def _fill_short_gaps(s: pd.Series, limit: int):
    """
    Interpolate NaN runs of at most `limit` steps; leave longer runs as NaN.

    pandas' own `interpolate(limit=...)` fills the first `limit` samples of an
    over-long run, which is the opposite of what is wanted here: a 24 h gap
    would silently acquire an hour of fabricated weather at each end. So the
    runs are identified first and long ones are restored to NaN afterwards.
    """
    missing = s.isna().to_numpy()
    if not missing.any():
        return s, missing, 0, 0, 0

    # label each maximal run of NaN, then measure the runs
    run_id = (missing != np.r_[False, missing[:-1]]).cumsum()
    run_of_nan = pd.Series(run_id[missing])
    lengths = run_of_nan.value_counts()
    long_ids = set(lengths[lengths > limit].index)

    keep_nan = np.zeros(len(s), bool)
    keep_nan[missing] = run_of_nan.isin(long_ids).to_numpy()

    # copy=True: pandas 3 hands back a read-only view otherwise
    filled = s.interpolate(limit_direction="both").to_numpy(copy=True)
    filled[keep_nan] = np.nan

    n_long = int(keep_nan.sum())
    return (pd.Series(filled, index=s.index, name=s.name), missing,
            int(missing.sum()) - n_long, n_long, len(long_ids))


def _segment(df: pd.DataFrame) -> list:
    """
    Split into contiguous 5-min runs with no NaN in the key columns.

    Weather columns are included in the key set, so a gap too long to
    interpolate ends the current segment. The Kalman filter then restarts with
    a fresh burn-in on the far side instead of propagating through fiction.
    """
    gap = df["timestamp"].diff().dt.total_seconds().to_numpy()
    bad = np.isnan(gap) | (np.abs(gap - DT) > 1.0)
    bad[0] = False

    key = ["T_i1", "T_i2", *Q_COLS, *WEATHER]
    nan_row = df[key].isna().any(axis=1).to_numpy()

    # a segment must break both entering and leaving a run of bad rows
    bad |= nan_row
    bad |= np.r_[False, nan_row[:-1]]

    edges = [0, *np.flatnonzero(bad).tolist(), len(df)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        # trim any leading bad rows the edge list carried in
        while a < b and nan_row[a]:
            a += 1
        if b - a >= MIN_SEGMENT:
            out.append((a, b))
    return out


def summarise(data: Dataset) -> None:
    df = data.df
    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    days = span.total_seconds() / 86400.0
    usable = sum(b - a for a, b in data.segments)

    print(f"  rows   : {len(df)}  span={span} ({days:.1f} days)")
    print(f"  segments: {len(data.segments)} (usable rows={usable}, "
          f"{100 * usable / len(df):.1f}% of record)")
    for a, b in data.segments:
        print(f"      {df.timestamp.iloc[a]:%Y-%m-%d %H:%M} .. "
              f"{df.timestamp.iloc[b - 1]:%Y-%m-%d %H:%M}  ({b - a} rows, "
              f"{(b - a) / 288:.1f} days)")

    print(f"  T_o    : {df.T_o.min():.1f} .. {df.T_o.max():.1f} degC")
    n_miss = int(df.T_o_missing.sum())
    if n_miss:
        by_month = df.groupby(df.timestamp.dt.to_period("M")).T_o_missing.mean()
        worst = by_month.idxmax()
        print(f"    ! {n_miss} rows ({100 * n_miss / len(df):.1f}%) missing "
              f"T_o; worst month {worst} at {100 * by_month.max():.0f}%")

    print(f"  v      : {df.v.min():.2f} .. {df.v.max():.2f} m/s "
          f"(p95={df.v.quantile(0.95):.2f})")
    if df.v.quantile(0.95) < 2.0:
        print("    ! low wind variance -> expect g1 to be weakly identified")
    print(f"  GHI    : {df.GHI.min():.0f} .. {df.GHI.max():.0f} W/m2")

    q = df["Q_total"]
    e_tot = q.sum() * DT / 3.6e6
    e_both = df.Q_dist_together.sum() * DT / 3.6e6
    print(f"  Q_dist : max={q.max():.0f} W, mean={q.mean():.0f} W "
          f"(over all rows, zeros included)")
    print(f"  energy : {e_tot:.0f} kWh total over {days:.1f} days "
          f"= {e_tot * 1000 / (days * 24):.0f} W average")
    print(f"           {100 * e_both / max(e_tot, 1e-9):.1f}% delivered with both "
          "zones calling (drives lam identifiability)")

    if data.has_setpoints:
        print(f"  setpts : PRESENT  T_i1_set {df.T_i1_set.min():.2f}"
              f"..{df.T_i1_set.max():.2f}, "
              f"T_i2_set {df.T_i2_set.min():.2f}..{df.T_i2_set.max():.2f} degC")
    if data.has_tw:
        print(f"  T_w    : present ({df.T_w.min():.1f} .. {df.T_w.max():.1f} degC)")
