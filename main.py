import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from house_parameters import HouseEnergyParamsComputer

logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s"
)

START_TIME = datetime(2025, 11, 1, tzinfo=ZoneInfo("America/New_York"))
END_TIME = datetime(2026, 3, 30, tzinfo=ZoneInfo("America/New_York"))

HOUSE_ALIASES = [
    "beech",
    "oak",
    "fir",
    "maple",
    "elm",
]

for house in HOUSE_ALIASES:
    print(f"\n[{house.capitalize()}]")
    try:
        h = HouseEnergyParamsComputer(house, START_TIME, END_TIME)
        energy_params = h.energy_fit_on_last_n_days(n=20)
        print(energy_params)
        rswt_params = h.rswt_fit_on_last_n_days(n=20)
        print(f"RSWT params:\n{rswt_params}")
        # h.trailing_n_day_fits(n=20)
        # h.sweep_n(min_n=5, max_n=200)
    except Exception as e:
        print(f"Error: {e}")
    
