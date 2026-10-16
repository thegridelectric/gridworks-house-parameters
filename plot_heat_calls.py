"""
Example heat-call Gantt plot (multi-zone layout with empty row slots).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from plots import plot_heat_calls

TZ = ZoneInfo("America/New_York")
DEFAULT_SAVEPATH = Path("results") / "heat_calls.pdf"


def _t(hour: int, minute: int = 0) -> datetime:
    return datetime(2025, 12, 25, hour, minute, tzinfo=TZ)


def _short_pulses(
    start: datetime, end: datetime, *, every_min: int = 8, on_min: int = 2
) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    t = start
    while t < end:
        t_on = t + timedelta(minutes=on_min)
        if t_on > end:
            break
        out.append((t, t_on))
        t += timedelta(minutes=every_min)
    return out


def main() -> None:
    window_start = _t(1, 0)
    window_end = _t(5, 30)

    zones = [
        (
            "Zone 1",
            _short_pulses(window_start, window_end),
            0,
        ),
        ("Zone 2", [(window_start, window_end)], 1),
        (
            "Zone 3",
            [
                (window_start + timedelta(minutes=22), window_start + timedelta(minutes=26)),
                (window_start + timedelta(hours=1, minutes=40), window_start + timedelta(hours=1, minutes=44)),
                (window_start + timedelta(hours=3, minutes=15), window_start + timedelta(hours=3, minutes=19)),
            ],
            2,
            3,
        ),
    ]
    plot_heat_calls(
        zones,
        title="Heat call activity",
        xlim=(window_start, window_end),
        savepath=DEFAULT_SAVEPATH,
        show=True,
    )
    print(f"Wrote {DEFAULT_SAVEPATH}")


if __name__ == "__main__":
    main()
