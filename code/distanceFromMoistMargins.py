# %%
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
from scipy import interpolate
import edgeFinder
from sklearn.cluster import DBSCAN

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


def rm_outlier(edge):

    points = list(zip(edge.longitude, edge))
    db = DBSCAN(eps=1.0, min_samples=5).fit(points)
    labels = db.labels_

    isolated_point_mask = labels == -1
    lat_south_connected = edge[~isolated_point_mask]

    return lat_south_connected


results = []

for time_i, time in enumerate(cwv_orcestra.time.values):
    cwv_i = cwv_orcestra.tcwv.isel(time=time_i)
    edges_i = all_time_results.isel(time=time_i)

    edge_results = []
    for edge_type in ["south", "north"]:
        edge_i = edges_i.sel(edge_type=edge_type).dropna(dim="longitude")
        edge_connected_i = rm_outlier(edge_i)

        if edge_connected_i.size == 0:
            new_field = xr.full_like(cwv_i, np.nan)
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
            new_field = cwv_i.latitude - latitude_edge_ds

        new_field = new_field.expand_dims(edge_type=[edge_type])
        edge_results.append(new_field)

    # First combine edges for this time step
    edges_for_time = xr.concat(edge_results, dim="edge_type")

    # Then add the time dimension
    edges_for_time = edges_for_time.expand_dims(time=[time])

    results.append(edges_for_time)


new_field_alltime = xr.concat(results, dim="time")
new_field_alltime.loc[dict(edge_type="south")] = -1 * new_field_alltime.sel(
    edge_type="south"
)
cwv_orcestra["distance_from_edge"] = new_field_alltime

# %%

fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
new_field_alltime.isel(time=1).sel(edge_type="south").plot()

# %%

for edge_type in ["south", "north"]:
    bins = np.arange(-10, 11, 0.1)
    tcwv_binned = cwv_orcestra.tcwv.groupby_bins(
        cwv_orcestra.distance_from_edge.sel(edge_type=edge_type), bins=bins
    ).mean(dim=["time", "latitude", "longitude"])

    tcwv_binned.plot(label=edge_type)

plt.legend()
# %%
