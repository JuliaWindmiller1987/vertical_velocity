#%%    

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal


#%%
cwv_thresh = 48
cwv_min = 42
perc_most = 25
#%%

cwv_orcestra = xr.open_dataset("/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc")
# %%

cwv_orcestra_mean = cwv_orcestra.tcwv.mean("time")

cwv_max = cwv_orcestra_mean.max("latitude")
cwv_max_lat = cwv_orcestra_mean.idxmax("latitude")
cwv_max_lat_smooth = cwv_max_lat.rolling(longitude=10, center=True).mean()

#%%

levels_cwv = np.sort(np.unique([cwv_thresh, 45, 50, 55]))

def find_edges_numpy(cwv, lat_cwv_max, latitudes, cwv_thresh=45, cwv_min=0):
    """
    Parameters:
        cwv (np.ndarray): 1D array of CWV values.
        lat_cwv_max (float): Latitude of CWV peak in the mean profile.
        latitudes (np.ndarray): 1D array of latitudes (same length as cwv).
    """
    if np.max(cwv) <= cwv_thresh:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peaks_i, _ = scipy.signal.find_peaks(cwv, height=cwv_thresh, prominence=2)
    if len(peaks_i) == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peak_lats = latitudes[peaks_i]
    closest_peak_idx = np.argmin(np.abs(peak_lats - lat_cwv_max))
    lat_peak = peak_lats[closest_peak_idx]

    moist_mask = cwv > cwv_min

    north_mask = (latitudes >= lat_peak) & moist_mask
    north_cwv = cwv[north_mask]
    north_lats = latitudes[north_mask]
    lat_north = np.min(north_lats[north_cwv <= cwv_thresh]) if np.any(north_cwv <= cwv_thresh) else np.nan

    south_mask = (latitudes <= lat_peak) & moist_mask
    south_cwv = cwv[south_mask]
    south_lats = latitudes[south_mask]
    lat_south = np.min(south_lats[south_cwv >= cwv_thresh]) if np.any(south_cwv >= cwv_thresh) else np.nan

    return np.array([lat_south, lat_north], dtype=np.float32)
#%%

latitudes_np = cwv_orcestra_mean.latitude.values

results = xr.apply_ufunc(
    find_edges_numpy,
    cwv_orcestra_mean,          # (latitude, longitude)
    cwv_max_lat_smooth,         # (longitude)
    kwargs={
        "latitudes": latitudes_np,
        "cwv_thresh": 45,
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

plt.figure(figsize = (20, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-65, -15, -5, 25], crs=ccrs.PlateCarree())
ax.coastlines(alpha=1.0)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha = 0.25)

cwv_orcestra_mean.plot(alpha = 0.75, cmap = 'Blues', vmin = 45, vmax = 70)
cwv_orcestra_mean.plot.contour(levels=levels_cwv, colors='k', linewidths=1, alpha = 0.5)


colEast = "C0"
colWest = "C1"

# Define box boundaries
lat_min, lat_max = 1, 22
lon_east_min, lon_east_max = -40, -19  # East box: 19W–40W
lon_west_min, lon_west_max = -61, -40  # West box: 61W–40W

# Define box corners
east_box_coords = [
    [lon_east_min, lat_min],
    [lon_east_max, lat_min],
    [lon_east_max, lat_max],
    [lon_east_min, lat_max]
]

west_box_coords = [
    [lon_west_min, lat_min],
    [lon_west_max, lat_min],
    [lon_west_max, lat_max],
    [lon_west_min, lat_max]
]

# Create and add East box
east_box = mpatches.Polygon(
    east_box_coords, closed=True,
    edgecolor=colEast, facecolor='none', linewidth=2,
    transform=ccrs.PlateCarree()
)
ax.add_patch(east_box)

# Create and add West box
west_box = mpatches.Polygon(
    west_box_coords, closed=True,
    edgecolor=colWest, facecolor='none', linewidth=2,
    transform=ccrs.PlateCarree()
)
ax.add_patch(west_box)

# Annotate boxes
ax.text((lon_east_min + lon_east_max)/2, lat_max + 1, 'East Box',
        color=colEast, transform=ccrs.PlateCarree(), ha='center')

ax.text((lon_west_min + lon_west_max)/2, lat_max + 1, 'West Box',
        color=colWest, transform=ccrs.PlateCarree(), ha='center')

plt.scatter(cwv_max_lat_smooth.longitude, cwv_max_lat_smooth)

plt.scatter(results.longitude, results.sel(edge_type=0))
plt.scatter(results.longitude, results.sel(edge_type=1))

# %%
