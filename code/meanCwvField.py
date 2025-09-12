import sys

sys.path.append("./code")

# %%

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from plotUtils import plot_cwv_field, add_east_west_boxes

import importlib
import edgeFinder

importlib.reload(edgeFinder)
from edgeFinder import find_edges_numpy

# %%
cwv_thresh = 50
cwv_min = 45
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

# %%

latitudes_np = cwv_orcestra_mean.latitude.values

results = xr.apply_ufunc(
    find_edges_numpy,
    cwv_orcestra_mean,  # (latitude, longitude)
    cwv_max_lat,  # (longitude)
    kwargs={
        "latitudes": latitudes_np,
        "cwv_thresh": cwv_thresh,
    },
    input_core_dims=[["latitude"], []],
    output_core_dims=[["edge_type"]],
    vectorize=True,
    output_dtypes=[np.float32],
    output_sizes={"edge_type": 2},
)

# %%

ax = plot_cwv_field(cwv_orcestra_mean)
add_east_west_boxes(ax)

plt.scatter(
    cwv_max_lat.longitude,
    cwv_max_lat,
    label="Latitude of max. CWV (smoothed)",
)
plt.scatter(results.longitude, results.sel(edge_type=0), label="Southern edge")
plt.scatter(results.longitude, results.sel(edge_type=1), label="Northern edge")
plt.legend()

sb.despine()

# plt.savefig("../figures/cwvMean.pdf", bbox_inches="tight")
