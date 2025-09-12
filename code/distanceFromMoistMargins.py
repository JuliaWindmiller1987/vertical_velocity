# %%
import sys

sys.path.append("./code")

# %%

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import importlib
from scipy import interpolate
import edgeFinder

importlib.reload(edgeFinder)
from edgeFinder import find_edges_numpy, rm_outlier

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
latitudes_np = cwv_orcestra_mean.latitude.values

# %%

all_time_results = xr.apply_ufunc(
    find_edges_numpy,
    cwv_orcestra.tcwv,  # (latitude, longitude)
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

all_time_results.compute()
all_time_results = all_time_results.assign_coords(
    edge_type=(
        "edge_type",
        ["south" if v == 0 else "north" for v in all_time_results.edge_type.values],
    )
)

# %%

results = []

for time_i, time in enumerate(cwv_orcestra.time.values):
    cwv_i = cwv_orcestra.tcwv.isel(time=time_i)
    edges_i = all_time_results.isel(time=time_i)

    edge_results = []
    for edge_type in ["south", "north"]:
        edge_i = edges_i.sel(edge_type=edge_type).dropna(dim="longitude")
        edge_connected_i = rm_outlier(edge_i)

        if edge_connected_i.size == 0:
            shifted_lat_field = xr.full_like(cwv_i, np.nan)
        else:
            spl, u = interpolate.splprep(
                [edge_connected_i], u=edge_connected_i.longitude, s=5
            )
            lat_fit = interpolate.splev(edge_connected_i.longitude, spl)[0]

            latitude_edge_ds = xr.DataArray(
                lat_fit,
                dims=("longitude",),
                coords={"longitude": edge_connected_i.longitude},
            )
            shifted_lat_field = cwv_i.latitude - latitude_edge_ds

        shifted_lat_field = shifted_lat_field.expand_dims(edge_type=[edge_type])
        edge_results.append(shifted_lat_field)

    edges_for_time = xr.concat(edge_results, dim="edge_type")
    edges_for_time = edges_for_time.expand_dims(time=[time])

    results.append(edges_for_time)

shifted_lat_field_alltime = xr.concat(results, dim="time")
shifted_lat_field_alltime.loc[dict(edge_type="south")] = (
    -1 * shifted_lat_field_alltime.sel(edge_type="south")
)

# %%

mask_north = shifted_lat_field_alltime.sel(
    edge_type="north"
) > shifted_lat_field_alltime.sel(edge_type="south")

shifted_lat_field_alltime.loc[dict(edge_type="north")] = shifted_lat_field_alltime.sel(
    edge_type="north"
).where(mask_north)
shifted_lat_field_alltime.loc[dict(edge_type="south")] = shifted_lat_field_alltime.sel(
    edge_type="south"
).where(~mask_north)

cwv_orcestra["distance_from_edge"] = shifted_lat_field_alltime

# %%

fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
shifted_lat_field_alltime.isel(time=1).sel(edge_type="south").plot()

# %%

results_bins = {}

for edge_type in ["south", "north"]:
    bins = np.arange(-5, 11, 0.1)
    tcwv_binned = cwv_orcestra.tcwv.groupby_bins(
        cwv_orcestra.distance_from_edge.sel(edge_type=edge_type), bins=bins
    ).mean(dim=["time", "latitude", "longitude"])

    results_bins[edge_type] = tcwv_binned

# %%
for edge_type in ["south", "north"]:

    results_bins[edge_type].plot(label=edge_type)

plt.legend()
sb.despine()
plt.title(" ")
plt.xlabel("distance from edge / °")
plt.ylabel("CWV / mm")

plt.axvline(0, color="k", linestyle=":")

# %%
