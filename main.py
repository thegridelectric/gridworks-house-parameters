import logging

from house_parameters import HouseEnergyParamsComputer

logging.basicConfig(
    level=logging.INFO, 
    format="%(levelname)s: %(message)s"
)

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
    h = HouseEnergyParamsComputer(house)
    h.trailing_n_day_fits(n=50)
    # h.sweep_n(min_n=5, max_n=200)
