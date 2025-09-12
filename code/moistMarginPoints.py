# %%
import sys

sys.path.append("./code")

# %%
import xarray as xr
import numpy as np
from plotUtils import plot_cwv_field, add_east_west_boxes
import matplotlib.pyplot as plt
import importlib
import edgeFinder

# %%
cwv_thresh = 50

# %%

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)
file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc"
cwv_orcestra = xr.open_dataset(data_path + file_name)
# %%

cwv_orcestra_mean = cwv_orcestra.tcwv.mean("time")

cwv_max = cwv_orcestra_mean.max("latitude")
cwv_max_lat = cwv_orcestra_mean.idxmax("latitude")

latitudes_np = cwv_orcestra_mean.latitude.values

cwv_orcestra_rolling = cwv_orcestra.tcwv.rolling(
    longitude=10, center=True, min_periods=1
).mean()

# %%

importlib.reload(edgeFinder)
from edgeFinder import find_cwv_center, find_edge_points

# %%

peak_lat = xr.apply_ufunc(
    find_cwv_center,
    cwv_orcestra_rolling,  # (latitude, longitude)
    cwv_max_lat,  # (longitude)
    kwargs={
        "cwv_thresh": cwv_thresh,
        "latitudes": latitudes_np,
    },
    input_core_dims=[["latitude"], []],
    output_core_dims=[[]],
    vectorize=True,
    output_dtypes=[np.float32],
)
# %%

cwv_north = cwv_orcestra.tcwv.where(cwv_orcestra.latitude >= peak_lat)
cwv_south = cwv_orcestra.tcwv.where(cwv_orcestra.latitude <= peak_lat)

# %%

time_ind = np.random.randint(len(cwv_orcestra.time))

ax = plot_cwv_field(
    cwv_orcestra.isel(time=time_ind).tcwv,
)
add_east_west_boxes(ax)
plt.scatter(peak_lat.isel(time=time_ind).longitude, peak_lat.isel(time=time_ind))


for cwv_field in [cwv_north, cwv_south]:

    ax = plot_cwv_field(
        find_edge_points(cwv_field.isel(time=time_ind), cwv_thresh, 1.0),
    )
    add_east_west_boxes(ax)
    plt.axvline(-55)

# %%
