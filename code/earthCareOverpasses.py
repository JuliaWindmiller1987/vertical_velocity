# %%

import datetime
from orcestra import get_flight_segments
import numpy as np
import pandas as pd

meta = get_flight_segments()

# %%

events = [
    {**e, "platform_id": platform_id, "flight_id": flight_id}
    for platform_id, flights in meta.items()
    for flight_id, flight in flights.items()
    for e in flight["events"]
]
# %%

dates_ec_coordination = [e["time"] for e in events if "ec_underpass" in e["kinds"]]


df = pd.DataFrame(
    {
        "date": [d.date() for d in dates_ec_coordination],
        "time": [d.time() for d in dates_ec_coordination],
    }
)
df.to_csv("./data/ecCoordinationDates.csv", index=False)
# %%
