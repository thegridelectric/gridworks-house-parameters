from house_parameters import HouseEnergyParamsComputer

HOUSE_ALIASES = [
    "beech",
    "oak",
    "fir",
    "maple_before",
    "maple_after",
    "elm"
]

for house in HOUSE_ALIASES:
    print(f"\nStarting analysis for {house}...")
    h = HouseEnergyParamsComputer(house)
    h.trailing_n_day_fits(n=20)
