#%%    

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from edgeFinder import find_edges_numpy
from plotUtils import plot_cwv_field, add_east_west_boxes


#%%
cwv_thresh = 50
cwv_min = 45
#%%

data_path = "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc"
cwv_orcestra = xr.open_dataset(data_path + file_name)
# %%

cwv_orcestra_mean = cwv_orcestra.tcwv.mean("time")

cwv_max = cwv_orcestra_mean.max("latitude")
cwv_max_lat = cwv_orcestra_mean.idxmax("latitude")
cwv_max_lat_smooth = cwv_max_lat.rolling(longitude=10, center=True).mean()


#%%

latitudes_np = cwv_orcestra_mean.latitude.values

results = xr.apply_ufunc(
    find_edges_numpy,
    cwv_orcestra_mean,          # (latitude, longitude)
    cwv_max_lat_smooth,         # (longitude)
    kwargs={
        "latitudes": latitudes_np,
        "cwv_thresh": cwv_thresh,
        "cwv_min": 0,
    },
    input_core_dims=[["latitude"], []],
    output_core_dims=[["edge_type"]],
    vectorize=True,
    output_dtypes=[np.float32],
    output_sizes={"edge_type": 2},
)

# %%

levels_cwv = np.sort(np.unique([cwv_thresh, 45, 50, 55]))
extent = [-65, -15, -5, 25]

ax = plot_cwv_field(cwv_orcestra_mean, levels=levels_cwv, extent=extent)
add_east_west_boxes(ax)

plt.scatter(cwv_max_lat_smooth.longitude, cwv_max_lat_smooth, label="Latitude of max. CWV (smoothed)")
plt.scatter(results.longitude, results.sel(edge_type=0), label="Southern edge")
plt.scatter(results.longitude, results.sel(edge_type=1), label="Northern edge")
plt.legend()

sb.despine()

plt.savefig("../figures/cwvMean.pdf", bbox_inches = "tight")

# %%

all_time_results = xr.apply_ufunc(
    find_edges_numpy,
    cwv_orcestra.tcwv,          # (latitude, longitude)
    cwv_max_lat_smooth,         # (longitude)
    kwargs={
        "latitudes": latitudes_np,
        "cwv_thresh": cwv_thresh,
        "cwv_min": 0,
    },
    input_core_dims=[["latitude"], []],
    output_core_dims=[["edge_type"]],
    vectorize=True,
    output_dtypes=[np.float32],
    output_sizes={"edge_type": 2},
)

all_time_results.compute()

# %%

time_ind = np.random.randint(len(cwv_orcestra.time))

test = cwv_orcestra.tcwv.isel(time=time_ind)

ax = plot_cwv_field(test, levels=levels_cwv, extent=extent)
add_east_west_boxes(ax)

test_results = all_time_results.isel(time=time_ind)

plt.scatter(cwv_max_lat_smooth.longitude, cwv_max_lat_smooth, label="Latitude of max. CWV (smoothed)")
plt.scatter(test_results.longitude, test_results.sel(edge_type=0), label="Southern edge")
plt.scatter(test_results.longitude, test_results.sel(edge_type=1), label="Northern edge")
plt.legend()
# %%


# %%
