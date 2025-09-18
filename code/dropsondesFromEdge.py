# %%
import sys

sys.path.append("./code")

# %%

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from plotUtils import plot_cwv_field, add_east_west_boxes, plot_map

# %%

lev4 = xr.open_dataset(
    "ipfs://bafybeiadtte5suphtlfq7r6vq5acuhz2n2ov3uaxjdbc7qw2n2m7nu4ktu", engine="zarr"
)

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)
file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19_w_edge_field.nc"
cwv_orcestra_with_edge = xr.open_dataset(data_path + file_name)

# %%

dic_subdomains = {
    "east": lev4.where(lev4.circle_lon >= -40, drop=True),
    "west": lev4.where(lev4.circle_lon < -40, drop=True),
}

# %%

fig, ax = plt.subplots(
    1, 1, figsize=(6, 4), subplot_kw={"projection": ccrs.PlateCarree()}, sharex=True
)

extent = [-65, -15, -5, 25]

ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines(alpha=1.0)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25)
ax.set_title(" ")

add_east_west_boxes(ax)

for i_key, key in enumerate(dic_subdomains.keys()):
    ds_sub = dic_subdomains[key]
    plt.scatter(ds_sub.circle_lon, ds_sub.circle_lat, s=3, color=f"C{i_key}")

# %%

results = []

for ind_ds in range(len(lev4.circle)):
    circle = lev4.isel(circle=ind_ds)
    d_circle_time = cwv_orcestra_with_edge.sel(
        time=circle.circle_time,
        longitude=360 + circle.circle_lon,
        latitude=circle.circle_lat,
        method="nearest",
    )

    results.append(d_circle_time)

# Combine everything along circle and edge_key
result_combined = xr.concat(results, dim="circle")
result_combined["wvel"] = lev4.wvel

# %%

for edge_t in ["north", "south"]:
    result_combined.min_distance_from_edge.sel(edge_type=edge_t).plot.hist(
        histtype="step",
        label=f"{len(result_combined.min_distance_from_edge.sel(edge_type=edge_t).dropna(dim="circle"))}",
    )

plt.legend()

# %%

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

linestyle = ["solid", "dotted"]

for i_domain, domain in enumerate(["west", "east"]):

    plt.sca(ax[i_domain])

    if domain == "east":
        result_domain = result_combined.where(
            result_combined.circle_lon >= -40, drop=True
        )

    elif domain == "west":
        result_domain = result_combined.where(
            result_combined.circle_lon < -40, drop=True
        )

    result_domain = result_domain.sel(altitude=slice(0, 12.5e3))

    for i_edge, edge_t in enumerate(["north", "south"]):
        distance_from_sel_edge = result_domain.min_distance_from_edge.sel(
            edge_type=edge_t
        )

        wvel_at_edge = result_domain.wvel.groupby_bins(
            distance_from_sel_edge,
            bins=np.arange(-3, 1.6, 1.5),
            labels=["center", "inside", "outside"],
        ).mean()

        for i_bin, bin in enumerate(wvel_at_edge.min_distance_from_edge_bins.values):
            wvel_at_edge.sel(min_distance_from_edge_bins=bin).plot(
                y="altitude",
                label=f"{bin} ITCZ ({edge_t})",
                color=f"C{i_bin}",
                linestyle=linestyle[i_edge],
            )

    plt.title(domain)
    plt.ylim(ymin=0)
    plt.axvline(0, linestyle="solid", linewidth=0.5, color="k")

sb.despine()

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.10),
)

# %%

ind_ds = 1

circle = ds_sub.isel(circle=ind_ds)

era_ind = cwv_orcestra_with_edge.sel(time=circle.circle_time, method="nearest")

a = era_ind.sel(longitude=360 + circle.circle_lon, method="nearest")
b = a.sel(latitude=circle.circle_lat, method="nearest")


fig, ax = plot_map()
era_ind.distance_south.plot()
era_ind.tcwv.plot.contour(levels=[46, 48, 50, 52])

plt.scatter(circle.circle_lon, circle.circle_lat)

# %%
