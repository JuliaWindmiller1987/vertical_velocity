# %%

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from plotUtils import add_east_west_boxes

# %%

lev4 = xr.open_dataset(
    "ipfs://bafybeiadtte5suphtlfq7r6vq5acuhz2n2ov3uaxjdbc7qw2n2m7nu4ktu", engine="zarr"
)

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

ind_ds = 0
key = "east"
ds_sub = dic_subdomains[key]

circle = ds_sub.isel(circle=ind_ds)
# %%
