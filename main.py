import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from house_parameters import HouseEnergyParamsComputer

logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s"
)

START_TIME = datetime(2026, 10, 10, tzinfo=ZoneInfo("America/New_York"))
END_TIME = datetime(2026, 10, 31, tzinfo=ZoneInfo("America/New_York"))

HOUSE_ALIASES = [
    "beech",
    # "oak",
    # "fir",
    # "maple",
    # "elm",
]

run_ms = int(time.time() * 1000)

for house in HOUSE_ALIASES:
    print(f"\n[{house.capitalize()}]")
    try:
        h = HouseEnergyParamsComputer(house, START_TIME, END_TIME, run_ms=run_ms)
        energy_params = h.get_energy_params()
        rswt_params = h.get_rswt_params(energy_params=energy_params)
        print(energy_params)
        print(rswt_params)
        # h.evaluate_trailing_forecast_performance(n=50)
        # h.sweep_n(min_n=5, max_n=200)
    except Exception as e:
        print(f"Error: {e}")
    
