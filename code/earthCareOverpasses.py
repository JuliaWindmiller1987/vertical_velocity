# %%

import datetime
from orcestra import get_flight_segments
import numpy as np
import pandas as pd

meta = get_flight_segments()

# %%

segments = [
    {**s, "platform_id": platform_id, "flight_id": flight_id}
    for platform_id, flights in meta.items()
    for flight_id, flight in flights.items()
    for s in flight["segments"]
]
# %%

dates_ec_coordination = np.unique(
    [s["start"].date() for s in segments if "ec_track" in s["kinds"]]
)

df = pd.DataFrame(dates_ec_coordination, columns=["date"])
df.to_csv("./data/ecCoordinationDates.csv", index=False)
# %%
