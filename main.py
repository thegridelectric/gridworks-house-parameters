import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from house_parameters import HouseEnergyParamsComputer

START_TIME = datetime(2025, 11, 1, tzinfo=ZoneInfo("America/New_York"))
END_TIME = datetime(2026, 3, 1, tzinfo=ZoneInfo("America/New_York"))

HOUSE_ALIASES = [
    "beech",
    "oak",
    "fir",
    "maple_before",
    "maple_after",
    "elm",
]

for house in HOUSE_ALIASES:
    print(f"\n[{house.capitalize()}]")
    try:
        h = HouseEnergyParamsComputer(house, START_TIME, END_TIME)
        energy_params = h.energy_fit_on_all_cleaned_data()
        h.plot_estimated_energy_vs_swt(energy_params)
        # rswt_params = h.rswt_fit_on_last_n_days(n=40)
        # print(f"Energy params:\n{energy_params}")
        # print(f"RSWT params:\n{rswt_params}")
    except Exception as e:
        print(f"Error: {e}")
    # h.trailing_n_day_fits(n=50)
    # h.sweep_n(min_n=5, max_n=200)
