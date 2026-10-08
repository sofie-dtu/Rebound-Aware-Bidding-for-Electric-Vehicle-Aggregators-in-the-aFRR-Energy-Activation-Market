#!/usr/bin/env python
"""
make_synthetic_charging.py
==========================

Generate a synthetic EV-charging dataset in the format the pipeline expects.

The EV charging data used in the paper is confidential; every other input is
public and can be re-downloaded (see data/README.md). So this script fills in
only the missing piece:

    Consumption/Consumption_perBox_15min.csv    15 min   chargeBoxId, power  [kW]
    Consumption/Consumption_agg_15min.csv       15 min   power               [kW]

Everything else -- aFRR activation prices, imbalance prices, day-ahead prices,
weather -- you supply yourself from the public sources. That makes a run on this
data *semi*-reproducible: the market side is the real market, and only the fleet
is invented. Results are qualitatively comparable to the paper and will not match
it numerically.

    # 1. put the public data in place (see data/README.md), then:
    python scripts/make_synthetic_charging.py
    python scripts/run_case_study.py --day 2025-12-18

By default the script reads the public files already in your data directory,
takes its date range and its temperature series from them, and writes the two
consumption files alongside them. Aligning to the real calendar matters: the
analogue-day search needs a charging profile for every candidate day, and a
fleet generated over a different date range would silently yield no candidates.

THE FLEET IS FICTIONAL. What it reproduces is the structure the model depends on:

  * Session-based charging, so an individual charger is idle most of the day
    and the fleet profile has a pronounced evening peak. Per-charger zeros
    drive the rebound headroom and the per-charger activation allocation.
  * Sessions that span midnight, so the fleet is never completely idle at
    04:00. A fleet that switches off overnight would make every day exceed
    `cons_zero_cap` and the scenario bank would come out empty.
  * A temperature response taken from YOUR weather file, so the
    temperature-based k-NN day selection has real signal to match on.
  * Occasional near-empty days, so the zero-heavy-day filters are exercised
    rather than dead code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Q_PER_DAY = 96

# Files the script can take its date range from, in order of preference.
# Imbalance and spot are stored timezone-naive at 15 min / 1 h; weather is
# stored UTC-aware and is the least convenient, so it comes last.
_REFERENCE_CANDIDATES = [
    ("Imbalance", "Imbalance_clean_{zone}.csv"),
    ("SpotPrice", "SpotPrice_clean_{zone}.csv"),
    ("Weather", "Weather_clean.csv"),
]


# ---------------------------------------------------------------------------
# Reading the public data
# ---------------------------------------------------------------------------
def _read_timestamped(path: Path) -> pd.DataFrame:
    """Read a cleaned CSV and return it with a sorted, tz-naive DatetimeIndex."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name} has no 'timestamp' column")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    # Stored as local wall time in some files and UTC in others; normalising to
    # tz-naive here keeps the generated index comparable to all of them.
    df = df.assign(timestamp=ts.dt.tz_convert(None)).dropna(subset=["timestamp"])
    return df.set_index("timestamp").sort_index()


def find_date_range(data_dir: Path, zone: str):
    """
    Infer the operating-day range from whichever public file is present.

    Returns (day_index, reference_name). Only whole days are kept: a partial
    first or last day would produce a short profile that make_daily_profiles
    turns into a row with fewer than 96 columns, which then propagates as NaN
    through the analogue search.
    """
    for folder, pattern in _REFERENCE_CANDIDATES:
        path = data_dir / folder / pattern.format(zone=zone)
        if not path.exists():
            continue
        df = _read_timestamped(path)
        if df.empty:
            continue

        first = df.index.min().normalize()
        last = df.index.max().normalize()
        # Drop the first/last day unless it is fully covered.
        if df.index.min() > first:
            first += pd.Timedelta(days=1)
        if df.index.max() < last + pd.Timedelta(days=1) - pd.Timedelta(minutes=15):
            last -= pd.Timedelta(days=1)
        if last < first:
            raise SystemExit(
                f"{path.name} covers less than one whole day; cannot infer a range.")

        return pd.date_range(first, last, freq="D"), path.name

    raise SystemExit(
        f"No public reference data found under {data_dir}.\n"
        f"  Looked for: "
        + ", ".join(f"{f}/{p.format(zone=zone)}" for f, p in _REFERENCE_CANDIDATES)
        + "\n  See data/README.md for the expected files and where to download them.\n"
          "  Or pass --start and --days to generate a fleet without reference data.")


def load_temperature(data_dir: Path, day_index: pd.DatetimeIndex):
    """
    Daily mean temperature from the real weather file, indexed by day.

    Falls back to a seasonal sine curve if no weather file is present, so the
    script still runs; the k-NN consumption selection is then matching on a
    smooth synthetic signal rather than real weather.
    """
    path = data_dir / "Weather" / "Weather_clean.csv"
    if path.exists():
        w = _read_timestamped(path)
        if "temperature_2m" in w.columns:
            daily = w["temperature_2m"].groupby(w.index.normalize()).mean()
            daily = daily.reindex(day_index).interpolate().bfill().ffill()
            if daily.notna().all():
                return daily, "Weather_clean.csv"

    doy = np.asarray(day_index.dayofyear)
    seasonal = 9.0 - 8.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    return pd.Series(seasonal, index=day_index), "seasonal fallback (no weather file)"


# ---------------------------------------------------------------------------
# The fleet model
# ---------------------------------------------------------------------------
def make_charging_day(day: pd.Timestamp, n_box: int, mean_temp: float,
                      rng: np.random.Generator,
                      carry_in: np.ndarray | None = None,
                      outage_rate: float = 0.06
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    One day of per-charger power in kW, with sessions spanning midnight.

    Returns (today, carry_out), both shape (96, n_box) in kW. `carry_in` is the
    previous day's overflow, added to this day; `carry_out` overflows into the
    next day.

    Each charger is idle most of the day and draws near-constant power during a
    session. Three arrival modes: overnight (plugged in during the evening and
    left), an evening top-up, and daytime workplace charging. Cold days lengthen
    sessions slightly, which is what gives the temperature-based analogue search
    something to match on.
    """
    # Two-day canvas, so a late session simply runs off the right-hand edge.
    canvas = np.zeros((2 * Q_PER_DAY, n_box))
    if carry_in is not None:
        canvas[:Q_PER_DAY] += carry_in

    is_weekend = day.dayofweek >= 5
    cold_factor = 1.0 + 0.012 * max(0.0, 12.0 - mean_temp)

    # Occasional site outage / data gap: a day where almost nothing is recorded.
    # These are the zero-heavy days the outlier filters exist to remove, so a
    # few keep that path exercised instead of dead.
    outage = rng.random() < outage_rate
    utilisation = 0.05 if outage else (0.42 if is_weekend else 0.58)

    for b in range(n_box):
        if rng.random() > utilisation:
            continue

        n_sessions = 1 if rng.random() > 0.18 else 2
        for _ in range(n_sessions):
            draw = rng.random()
            if draw < 0.45:                       # overnight, long dwell
                arrival_h = rng.normal(20.5, 2.4)
                dur_q = rng.lognormal(np.log(38), 0.45)    # ~9.5 h median
            elif draw < 0.75:                     # evening top-up
                arrival_h = rng.normal(17.5, 2.0)
                dur_q = rng.lognormal(np.log(11), 0.55)    # ~2.75 h median
            else:                                 # daytime
                arrival_h = rng.normal(9.5, 2.2)
                dur_q = rng.lognormal(np.log(14), 0.55)    # ~3.5 h median

            arrival_q = int(np.clip(arrival_h * 4, 0, 2 * Q_PER_DAY - 3))
            dur_q = int(np.clip(dur_q * cold_factor, 2, 2 * Q_PER_DAY - arrival_q))
            rated = rng.choice([3.7, 7.4, 11.0, 22.0], p=[0.20, 0.42, 0.30, 0.08])

            # Near-constant draw tapering over the final third of the session
            # (constant-voltage phase, then an idle car still plugged in).
            profile = np.full(dur_q, rated) * (1.0 + rng.normal(0, 0.03, dur_q))
            taper = max(2, dur_q // 3)
            profile[-taper:] *= np.linspace(0.75, 0.0, taper)

            canvas[arrival_q:arrival_q + dur_q, b] += np.maximum(profile, 0.0)

    return canvas[:Q_PER_DAY].round(3), canvas[Q_PER_DAY:].round(3)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def generate(out_dir: Path, day_index: pd.DatetimeIndex, temps: pd.Series,
             n_box: int, seed: int, outage_rate: float) -> None:
    rng = np.random.default_rng(seed)
    n_days = len(day_index)

    quarters = pd.date_range(day_index[0], periods=n_days * Q_PER_DAY, freq="15min")
    box_ids = [f"CB{b:04d}_1" for b in range(n_box)]

    # Burn-in: one discarded day so the first real day inherits overnight
    # sessions instead of starting from an artificially empty fleet.
    _, carry = make_charging_day(day_index[0] - pd.Timedelta(days=1), n_box,
                                 float(temps.iloc[0]), rng, outage_rate=outage_rate)

    agg_values, box_blocks = [], []
    for d, day in enumerate(day_index):
        power_day, carry = make_charging_day(
            day, n_box, float(temps.iloc[d]), rng,
            carry_in=carry, outage_rate=outage_rate)
        agg_values.append(power_day.sum(axis=1))
        box_blocks.append(power_day)

        if (d + 1) % 25 == 0 or d == n_days - 1:
            print(f"  day {d + 1}/{n_days}")

    # Long format: one row per (timestamp, chargeBoxId).
    stacked = np.concatenate(box_blocks)                    # (n_days*96, n_box)
    per_box = pd.DataFrame({
        "timestamp": np.repeat(quarters, n_box),
        "chargeBoxId": np.tile(box_ids, len(quarters)),
        "power": stacked.ravel(),
    }).set_index("timestamp")

    agg = pd.DataFrame({"power": np.concatenate(agg_values).round(3)}, index=quarters)

    folder = out_dir / "Consumption"
    folder.mkdir(parents=True, exist_ok=True)
    targets = {
        folder / "Consumption_perBox_15min.csv": per_box,
        folder / "Consumption_agg_15min.csv": agg,
    }

    print()
    for path, frame in targets.items():
        frame.to_csv(path, index_label="timestamp")
        print(f"  wrote {path.name:34s} {len(frame):>9,} rows  "
              f"{path.stat().st_size / 1e6:6.1f} MB")

    # --- Summary, including the checks that matter to the pipeline ----------
    agg_mw = agg["power"].to_numpy() / 1000.0
    per_day_zero = (agg["power"].to_numpy() == 0).reshape(n_days, Q_PER_DAY).mean(axis=1)
    n_dropped = int((per_day_zero > 0.15).sum())

    print(f"\nSummary")
    print(f"  Days:                  {n_days}  "
          f"({day_index[0].date()} to {day_index[-1].date()})")
    print(f"  Chargers:              {n_box}")
    print(f"  Fleet load:            mean {agg_mw.mean():.3f} MW, "
          f"peak {agg_mw.max():.3f} MW")
    print(f"  Zero quarters (fleet): {100 * (agg_mw == 0).mean():.2f}%")
    print(f"  Days over cons_zero_cap=0.15: {n_dropped} "
          f"({n_days - n_dropped} usable)")

    if n_days - n_dropped < 25:
        print("\n  [warn] fewer than 25 usable days: the analogue-day search may not\n"
              "         find k candidates. Use a longer market-data range.")


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--data-dir", type=Path, default=None,
                   help="directory holding the public data; the consumption "
                        "files are written here too (default: AFRR_DATA_DIR, "
                        "else data/)")
    p.add_argument("--out", type=Path, default=None,
                   help="write the consumption files somewhere else instead")
    p.add_argument("--zone", default="DK1")
    p.add_argument("--chargers", type=int, default=100,
                   help="number of charge boxes (default: 100). Below about 50 "
                        "the fleet is sparse enough that many days exceed "
                        "cons_zero_cap and get dropped from the scenario bank.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outage-rate", type=float, default=0.06,
                   help="fraction of days that are near-empty (default: 0.06); "
                        "0 disables them")

    g = p.add_argument_group(
        "without reference data",
        "Normally the date range is taken from your public market data. These "
        "override it when you have none yet.")
    g.add_argument("--start", default=None, help="first day, e.g. 2025-10-01")
    g.add_argument("--days", type=int, default=None, help="number of days")

    args = p.parse_args(argv)

    data_dir = args.data_dir
    if data_dir is None:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from config import DATA_DIR
            data_dir = DATA_DIR
        except Exception:
            data_dir = Path("data")
    data_dir = Path(data_dir)
    out_dir = Path(args.out) if args.out else data_dir

    # --- Date range: explicit, or inferred from the public data -------------
    if args.start or args.days:
        if not (args.start and args.days):
            p.error("--start and --days must be given together")
        day_index = pd.date_range(args.start, periods=args.days, freq="D")
        reference = "--start / --days"
    else:
        day_index, reference = find_date_range(data_dir, args.zone)

    temps, temp_source = load_temperature(data_dir, day_index)

    print(f"Data directory : {data_dir}")
    print(f"Date range     : {day_index[0].date()} to {day_index[-1].date()} "
          f"({len(day_index)} days, from {reference})")
    print(f"Temperature    : {temp_source}")
    print(f"Fleet          : {args.chargers} chargers, seed {args.seed}\n")

    if len(day_index) < 25:
        print("[warn] fewer than 25 days: the analogue-day search needs enough\n"
              "       candidates to pick k from.\n")

    # A small fleet is genuinely sparse: individual sessions no longer average
    # out, so whole quarters fall to zero and those days are dropped by
    # cons_zero_cap. Measured over 60 days: 100 chargers loses 1 day, 50 loses
    # 8, 20 loses 10, and 10 chargers loses 35.
    if args.chargers < 50:
        print(f"[warn] {args.chargers} chargers is a sparse fleet; expect days to be\n"
              "       dropped for having too many zero quarters. Use --chargers 100\n"
              "       or more unless you are deliberately testing a small fleet.\n")

    generate(out_dir, day_index, temps, args.chargers, args.seed, args.outage_rate)

    if out_dir == data_dir:
        print(f"\nThe pipeline can now run against {data_dir}:")
        print(f"  export AFRR_DATA_DIR={Path(data_dir).resolve()}")
        print(f"  python scripts/run_case_study.py --day {day_index[len(day_index) // 2].date()}")


if __name__ == "__main__":
    main()
