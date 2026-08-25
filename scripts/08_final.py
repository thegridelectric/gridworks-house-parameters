"""
Head-to-head: every candidate hour-ahead Q_dist predictor, on the same windows.

Includes the deployed alpha/beta/gamma coefficients as the incumbent, both
as-is and with the offset refitted, since a constant offset is trivially
fixable and leaving it in would flatter everything else.

Protocol is identical for every candidate: fit on the first 70%, choose among
variants on the next 10%, report on the last 20%. Windows are intersected on
timestamp, so no model is scored on hours another one skipped.

The last section is the one that matters most. It asks whether the RC model
carries information the simple features do not, by regressing actual energy on
the simple features PLUS the RC prediction. If the RC term earns a meaningful
coefficient and lowers the error, the physics is contributing something the
regression cannot see. If it does not, the extra machinery is not paying for
itself on this house.

    uv run -u scripts/08_final.py
"""

import json
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house.data import load_data
from house.forecast import forecast
from house.simple import FEATURES, Linear, design, hourly_frame, score_energy
from house.thermostat import calibrate

RESULTS = Path(__file__).resolve().parents[1] / "results"
HERE = Path(__file__).resolve().parent
BEECH = (Path(__file__).resolve().parents[1] / "data" /
         "beech_electricity_use_2025-11-01-00-00-2026-05-01-00-00.csv")
SPLITS = ("fit", "sel", "test")


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def load_module(name, fname):
    spec = spec_from_file_location(name, HERE / fname)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rc_candidates(ef, de):
    """Every RC fit that has been produced, with its controller settings."""
    out = []
    p6 = RESULTS / "06_energy_fit.json"
    if p6.exists():
        b6 = json.loads(p6.read_text())
        variants = dict(ef.variants())
        for key in ("tau_e bounded + free q", "tau_e + tau_m bounded + free q"):
            if key in b6:
                out.append((f"4R3C likelihood-fit ({key})", variants[key],
                            b6[key]["params"], None, None))
    p7 = RESULTS / "07_direct_energy.json"
    if p7.exists():
        b7 = json.loads(p7.read_text())
        out.append(("4R3C energy-optimised", de.MODEL, b7["params"],
                    b7.get("on_power"), None))
    p9 = RESULTS / "09_supply.json"
    if p9.exists():
        b9 = json.loads(p9.read_text())
        # JSON round-trips dict keys as strings; the controller indexes by int
        sup = {"T_w": b9["supply"]["T_w"],
               "K_w": {int(k): v for k, v in b9["supply"]["K_w"].items()}}
        out.append(("4R3C energy-optimised + supply law", de.MODEL,
                    b9["params"], None, sup))
    return out


def main() -> None:
    ef = load_module("ef", "06_energy_fit.py")
    de = load_module("de", "07_direct_energy.py")

    data = load_data(verbose=False)
    df = data.df
    parts, masks = ef.three_way(data)
    cal = calibrate(df[masks["fit"]].reset_index(drop=True))

    # ---- RC forecasts on all three splits --------------------------------
    rule("RC CANDIDATES")
    rc = {}
    for label, model, p, pw, sup in rc_candidates(ef, de):
        zones = cal if not pw else {
            1: replace(cal[1], on_power=float(pw["1"])),
            2: replace(cal[2], on_power=float(pw["2"]))}
        per_split = {}
        for k in SPLITS:
            v = forecast(model, p, zones, df, parts[k], masks[k],
                         nbins=6, burn=288, supply=sup)
            per_split[k] = v.set_index("t")
        rc[label] = per_split
        s = score_energy(per_split["sel"].pred, per_split["sel"].actual)
        print(f"  {label:<48s} selection MAE {s['mae_kwh']:.3f}", flush=True)

    if not rc:
        sys.exit("no RC results found; run 06/07/09 first")

    ref = next(iter(rc.values()))["test"]
    actual = ref.actual

    # ---- simple models ----------------------------------------------------
    # same segments and burn-in as the RC forecast, so the window grids match
    hs = {k: hourly_frame(df, cal, segments=parts[k], burn=288).set_index("t")
          for k in SPLITS}

    models = {n: Linear(n, f).fit(hs["fit"])
              for n, f in FEATURES.items() if f is not None}
    rule("SIMPLE MODELS")
    print(f"  fitted on {len(hs['fit'])} hours\n")
    best_simple = None
    for n, m in models.items():
        sm = score_energy(m.predict(hs["sel"]), hs["sel"].actual)["mae_kwh"]
        print(f"  {n:<32s} sel MAE {sm:.3f}   {m.describe()}", flush=True)
        if best_simple is None or sm < best_simple[0]:
            best_simple = (sm, n, m)
    print(f"\n  -> selected: {best_simple[1]}", flush=True)

    # ---- assemble predictions on the common test windows ------------------
    preds = {label: d["test"].pred for label, d in rc.items()}
    preds["persistence (last hour)"] = ref.persistence
    h_test = hs["test"].reindex(actual.index)
    for n, m in models.items():
        preds[n] = pd.Series(m.predict(h_test.fillna(0.0)),
                             index=h_test.index).where(h_test.actual.notna())

    if BEECH.exists():
        e = pd.read_csv(BEECH)
        e["t"] = pd.to_datetime(e.hour_start)
        e = e.drop_duplicates("t").set_index("t").sort_index()
        for k in SPLITS:
            hs[k] = hs[k].join(e[["alpha", "beta"]], how="left")
        raw = {k: (hs[k].alpha + hs[k].beta * (hs[k].T_o * 9 / 5 + 32))
               .clip(lower=0) for k in SPLITS}
        shift = float((hs["fit"].actual - raw["fit"]).mean())
        preds["deployed alpha/beta/gamma"] = raw["test"].reindex(actual.index)
        preds["deployed a/b/g, offset refitted"] = (
            (raw["test"] + shift).clip(lower=0).reindex(actual.index))
        print(f"  incumbent offset correction: {shift:+.2f} kWh", flush=True)

    # ---- does the RC model add anything the regression cannot see? --------
    rule("HYBRID: simple features + RC prediction")
    feats = best_simple[2].feats
    for label, d in rc.items():
        frames = {}
        for k in SPLITS:
            j = hs[k].join(d[k][["pred"]].rename(columns={"pred": "rc"}),
                           how="inner").dropna(subset=["rc", "actual"])
            frames[k] = j
        if len(frames["fit"]) < 100:
            continue
        X = np.column_stack([np.ones(len(frames["fit"])),
                             design(frames["fit"], feats),
                             frames["fit"].rc.to_numpy()])
        coef, *_ = np.linalg.lstsq(X, frames["fit"].actual.to_numpy(),
                                   rcond=None)

        def apply(fr):
            Xf = np.column_stack([np.ones(len(fr)), design(fr, feats),
                                  fr.rc.to_numpy()])
            return np.clip(Xf @ coef, 0.0, None)

        s_sel = score_energy(apply(frames["sel"]), frames["sel"].actual)
        s_test = score_energy(apply(frames["test"]), frames["test"].actual)
        print(f"  {label}", flush=True)
        print(f"    RC coefficient {coef[-1]:+.3f}   "
              f"sel MAE {s_sel['mae_kwh']:.3f}   "
              f"test MAE {s_test['mae_kwh']:.3f}  r {s_test['r']:.3f}",
              flush=True)
        preds[f"hybrid ({label})"] = pd.Series(
            apply(frames["test"]), index=frames["test"].index)

    # ---- blend: the two disagree partly independently ---------------------
    # The RC model and the regression correlate about 0.86 with each other, so
    # averaging them cancels some error. The weight is chosen on the selection
    # split, never on the test set.
    rule("BLEND: w * RC + (1-w) * best simple model")
    simple_by_split = {k: pd.Series(best_simple[2].predict(hs[k]),
                                    index=hs[k].index) for k in SPLITS}
    print(f"  {'w':>5s} {'sel MAE':>9s} {'test MAE':>9s}")
    for label, d in rc.items():
        frames = {}
        for k in ("sel", "test"):
            frames[k] = pd.DataFrame({
                "actual": d[k].actual, "rc": d[k].pred,
                "simple": simple_by_split[k]}).dropna()
        chosen = None
        for w in np.arange(0.0, 1.01, 0.1):
            sm = score_energy(w * frames["sel"].rc
                              + (1 - w) * frames["sel"].simple,
                              frames["sel"].actual)["mae_kwh"]
            if chosen is None or sm < chosen[0]:
                chosen = (sm, w)
        w = chosen[1]
        bl = w * frames["test"].rc + (1 - w) * frames["test"].simple
        st = score_energy(bl, frames["test"].actual)
        print(f"  {label}: w={w:.1f} (sel {chosen[0]:.3f}) -> "
              f"test MAE {st['mae_kwh']:.3f}  r {st['r']:.3f}", flush=True)
        preds[f"BLEND {w:.0%} ({label})"] = bl

    # ---- final table ------------------------------------------------------
    rows = []
    for name, series in preds.items():
        s = pd.Series(series).reindex(actual.index)
        m = s.notna() & actual.notna()
        if m.sum() < 20:
            continue
        rows.append({"model": name, **score_energy(s[m], actual[m])})
    t = pd.DataFrame(rows).sort_values("mae_kwh")

    rule("TEST SET  (final 20%, never used for fitting or selection)")
    print(f"  mean actual {actual.mean():.3f} kWh/hour over "
          f"{len(actual)} windows\n")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    t.to_csv(RESULTS / "08_final.csv", index=False)
    _plot(preds, actual, t, RESULTS / "08_final.png")
    print(f"\n  -> {RESULTS / '08_final.csv'}", flush=True)


def _plot(preds, actual, table, path):
    top = table.head(4).model.tolist()
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    hi = float(actual.max())
    for a, name in zip(ax.ravel(), top):
        s = pd.Series(preds[name]).reindex(actual.index)
        m = s.notna()
        a.plot([0, hi], [0, hi], "k--", lw=1)
        a.scatter(actual[m], s[m], s=14, alpha=0.5)
        row = table[table.model == name].iloc[0]
        a.set_title(f"{name}\nMAE {row.mae_kwh:.3f} kWh, r={row.r:.3f}",
                    fontsize=9)
        a.set_xlabel("actual [kWh]")
        a.set_ylabel("predicted [kWh]")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  plot -> {path}", flush=True)


if __name__ == "__main__":
    main()
