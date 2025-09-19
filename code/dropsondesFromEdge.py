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

print("Removes flag distance fields")
result_combined = result_combined.where(result_combined.flag == "good", drop=True)
# %%

result_combined.distance.plot.hist(bins=20, density=True)

(result_combined.distance).quantile(0.77)

# %%

results_binned = result_combined[["tcwv", "wvel", "distance"]].groupby_bins(
    result_combined.distance,
    bins=np.arange(-5, 2.6, 2.5),
)
results_binned_mean = results_binned.mean()
results_binned_count = results_binned.count()
print(results_binned_count.tcwv.values, results_binned_count.tcwv.sum().values)

for i_bin, bin in enumerate(results_binned_mean.distance_bins):

    bin_count = results_binned_count.sel(distance_bins=bin)
    wvel_bin = results_binned_mean.sel(distance_bins=bin).wvel
    wvel_bin = wvel_bin.where(bin_count.wvel / bin_count.distance > 0.9, drop=True)

    bin_str = f"{bin.values} (#{int(bin_count.distance.values)}): {wvel_bin.mean().values*1000:.1f} mm/s"
    wvel_bin.plot(y="altitude", label=bin_str)

plt.ylim(0, 12.5e3)
plt.xlim(-0.025, 0.025)
sb.despine()
plt.legend(loc=3, fontsize=8)
plt.title("ORCESTRA dropsonde circles \n binned by distance to edge")
plt.tight_layout()
plt.savefig("../figures/dropsonde_distance_edge.png")
# %%
# Example plot

ind_ds = 1

circle = ds_sub.isel(circle=ind_ds)

era_ind = cwv_orcestra_with_edge.sel(time=circle.circle_time, method="nearest")

a = era_ind.sel(longitude=360 + circle.circle_lon, method="nearest")
b = a.sel(latitude=circle.circle_lat, method="nearest")


fig, ax = plot_map()
era_ind.distance.plot(cmap="RdBu_r", alpha=0.75, vmin=-7.5, vmax=7.5)
era_ind.tcwv.plot.contour(levels=[46, 48, 50, 52])

plt.scatter(circle.circle_lon, circle.circle_lat)

# %%
