"""
Plot hourly outside air temperature from loaded (uncleaned) house data.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from house_parameters import HouseEnergyParamsComputer
from plots import plot_oat_timeseries_with_swt_markers

START_TIME = datetime(2025, 11, 1, tzinfo=ZoneInfo("America/New_York"))
END_TIME = datetime(2026, 3, 1, tzinfo=ZoneInfo("America/New_York"))

HOUSE_ALIAS = "oak"
DEFAULT_SAVEPATH = Path("results") / f"{HOUSE_ALIAS}_oat_timeseries.pdf"


def main() -> None:
    computer = HouseEnergyParamsComputer(
        HOUSE_ALIAS,
        START_TIME,
        END_TIME,
        prepare=False,
    )
    df = computer.df
    plot_oat_timeseries_with_swt_markers(
        df["hour_start"],
        df["oat_f"].to_numpy(),
        [],
        savepath=DEFAULT_SAVEPATH,
    )
    print(f"Wrote {DEFAULT_SAVEPATH}")


if __name__ == "__main__":
    main()
