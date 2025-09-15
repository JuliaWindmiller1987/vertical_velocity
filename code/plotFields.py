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

plt.savefig("./figures/cwvMean.pdf", bbox_inches="tight")

# %%


cwv_orcestra_with_edge = xr.open_dataset(
    data_path + file_name.split(".")[0] + "_w_edge_field" + ".nc"
)

fig, axes = plt.subplots(
    3, 1, figsize=(6, 10), subplot_kw={"projection": ccrs.PlateCarree()}, sharex=True
)

extent = [-65, -15, -5, 25]
time_ind = 20

ax = axes[0]
plt.sca(ax)

cbar_kwargs = {
    "shrink": 0.7,
    "aspect": 30,
    "pad": 0.1,
}

plot_cwv_field(
    cwv_orcestra_with_edge.tcwv.isel(time=time_ind),
    levels=[45, 55],
    ax=ax,
    cbar_kwargs=cbar_kwargs,
)
cwv_orcestra_with_edge.tcwv.isel(time=time_ind).plot.contour(
    levels=[cwv_thresh], colors="k"
)
add_east_west_boxes(ax)

for i_edge, edge in enumerate(["south", "north"]):

    ax = axes[i_edge + 1]
    plt.sca(ax)

    cwv_orcestra_with_edge.sel(edge_type=edge)["distance_from_edge"].isel(
        time=time_ind
    ).plot(
        cmap="BrBG",
        add_colorbar=True,
        vmin=-15,
        vmax=15,
        levels=np.arange(-15, 15, 1),
        cbar_kwargs={**cbar_kwargs, "label": f"distance from {edge}ern edge \n [°]"},
    )

    cwv_orcestra_with_edge.tcwv.isel(time=time_ind).plot.contour(
        levels=[cwv_thresh], colors="k"
    )

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=1.0)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25)
    ax.set_title(" ")

    add_east_west_boxes(ax)

plt.tight_layout()
plt.savefig("./figures/example_edge.png")
# %%

results_bins = {}

for edge_type in ["south", "north"]:
    bins = np.arange(-5, 11, 0.1)
    tcwv_binned = cwv_orcestra_with_edge.tcwv.groupby_bins(
        cwv_orcestra_with_edge.distance_from_edge.sel(edge_type=edge_type), bins=bins
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
