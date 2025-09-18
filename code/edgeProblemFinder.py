# %%
import sys

sys.path.append("./code")
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.cluster import DBSCAN
import cartopy.crs as ccrs


import importlib
import edgeFinder

importlib.reload(edgeFinder)
from edgeFinder import find_edges_numpy, rm_outlier, find_crossing

from plotUtils import plot_cwv_field, add_east_west_boxes, plot_map

# %%

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)

file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc"
cwv_orcestra = xr.open_dataset(data_path + file_name)

file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19_w_edge_field.nc"
cwv_orcestra_with_edge = xr.open_dataset(data_path + file_name)

cwv_orcestra_mean = cwv_orcestra_with_edge.tcwv.mean("time")

cwv_max = cwv_orcestra_mean.max("latitude")
# cwv_max_lat = cwv_orcestra_mean.idxmax("latitude")
cwv_max_lat = (
    cwv_orcestra_with_edge.sel(time="2024-09-07").tcwv.mean("time").idxmax("latitude")
)
# %%

era_ind = cwv_orcestra_with_edge.sel(
    time="2024-09-07T15:00:00.000000000", method="nearest"
)
plot_cwv_field(era_ind.tcwv)

# %%

lon_error = era_ind.longitude.where(
    np.isnan(era_ind.sel(edge_type="south").isel(latitude=-1).min_distance_from_edge),
    drop=True,
).values
# %%
lon_ind = lon_error[1]
# %%

tcwv_ind = era_ind.tcwv.sel(longitude=lon_ind).values
lat_ind = era_ind.latitude.values


edges_ind = find_edges_numpy(
    tcwv_ind, float(cwv_max_lat.sel(longitude=lon_ind)), lat_ind, cwv_thresh=cwv_thresh
)

edges_ind
# %%
####################

importlib.reload(edgeFinder)
from edgeFinder import find_edges_numpy, rm_outlier, filter_field_with_dbscan

era_ind = cwv_orcestra_with_edge.sel(
    time="2024-08-01T12:00:00.000000000", method="nearest"
)

test_field = era_ind.sel(longitude=slice(299, 320))
ax = plot_cwv_field(test_field.tcwv)
add_east_west_boxes(ax)

snapshot_mean = test_field.tcwv.mean("longitude")

# plt.figure()
# snapshot_mean.plot()
# plt.axvline(edges_ind[0])
# plt.axvline(edges_ind[1])


cwv_thresh = 48

if snapshot_mean.max() < cwv_thresh:
    print("Need to break the loop and return none")

cwv_max_lat = snapshot_mean.idxmax("latitude")

cwv_crossings = find_crossing(snapshot_mean, snapshot_mean.latitude, cwv_thresh)
cwv_lat_north = cwv_crossings[cwv_crossings.latitude > cwv_max_lat].min().values
cwv_lat_south = cwv_crossings[cwv_crossings.latitude < cwv_max_lat].max().values


for edges_where in [0, 1]:
    edges_i = []

    for lon_ind in test_field.tcwv.longitude:
        tcwv_ind = test_field.tcwv.sel(longitude=lon_ind).values
        lat_ind = test_field.latitude.values

        edges_ind = find_edges_numpy(
            tcwv_ind,
            float(cwv_max_lat),
            cwv_lat_south,
            cwv_lat_north,
            lat_ind,
            cwv_thresh=cwv_thresh,
        )

        edges_i.append(edges_ind[edges_where])  # change here for north or south

    edges_i = xr.DataArray(
        edges_i,
        dims=("longitude",),
        coords={"longitude": test_field.tcwv.longitude - 360},
    )

    edge_i = edges_i.dropna(dim="longitude")
    edge_connected_i = rm_outlier(edge_i, min_cluster_size=12)
    edge_connected_i.plot(c="k")

    s = 5

    spl = interpolate.make_splrep(edge_connected_i.longitude, edge_connected_i, s=s)
    lat_fit = spl(edge_connected_i.longitude)

    plt.scatter(edge_connected_i.longitude, lat_fit, label=s)
    edge_connected_i.plot(c="k")
# %%

era_ind = cwv_orcestra_with_edge.sel(
    time="2024-08-23T17:00:00.000000000", method="nearest"
)

# dic_lons = {"east": [299, 320], "west": [320, 341]}

# fig, ax = plt.subplots(1,2)

# for i_region, region in enumerate(["east", "west"]):
# plt.sca(ax[i_region])
# lon_min, lon_max = dic_lons[region]

test_field = era_ind  # .sel(longitude=slice(lon_min, lon_max))
filtered_mask, labels = filter_field_with_dbscan(
    test_field.tcwv, 48, eps=2, min_cluster_size=10**2 * 4 * 4
)

plt.imshow(filtered_mask)


plot_cwv_field(era_ind.tcwv)
# %%

test_field.tcwv
# %%

from scipy.ndimage import label, binary_erosion, binary_dilation, distance_transform_edt


def keep_largest_connected_component(binary_field):
    labeled_array, num_features = label(binary_field)
    if num_features == 0:
        return np.zeros_like(binary_field)

    unique_all, counts_all = np.unique(labeled_array, return_counts=True)
    unique = unique_all[unique_all != 0]
    counts = counts_all[unique_all != 0]

    largest_label = unique[np.argmax(counts)]
    return np.where(labeled_array == largest_label, 1, 0)


# %%


def pick_single_key_region(
    binary_field, structure1, iterations1, structure2, iterations2
):

    eroded_dilated = binary_erosion(
        binary_field, structure=structure1, iterations=iterations1
    ).astype(int)

    eroded_dilated = keep_largest_connected_component(eroded_dilated)

    eroded_dilated = binary_dilation(
        eroded_dilated, structure=structure1, iterations=iterations1
    ).astype(int)

    eroded_dilated = binary_dilation(
        eroded_dilated, structure=structure2, iterations=iterations2
    ).astype(int)

    eroded_dilated = binary_erosion(
        eroded_dilated, structure=structure2, iterations=iterations2
    ).astype(int)

    return eroded_dilated


cwv_thresh = 50
res_in_deg = 0.25

iterations1 = 4
structure1 = np.zeros((3, 3))
structure1[1, :] = 1

iterations2 = 4
structure2 = np.ones((3, 3))

distance_A = []
key_cwv_region_A = []

for i_time, time in enumerate(cwv_orcestra.time.values):

    cwv_field = cwv_orcestra.sel(time=time).tcwv.values

    key_cwv_region = pick_single_key_region(
        np.where(cwv_field > cwv_thresh, 1, 0),
        structure1,
        iterations1,
        structure2,
        iterations2,
    )

    distance_inside = -1 * distance_transform_edt(key_cwv_region)
    distance_outside = distance_transform_edt(np.where(key_cwv_region == 0, 1, 0))

    distance = (distance_outside + distance_inside) * res_in_deg

    distance_A.append(distance)
    key_cwv_region_A.append(key_cwv_region)

# %%

distance_ds = xr.Dataset(
    data_vars={
        "largest_cluster": (("time", "latitude", "longitude"), key_cwv_region_A),
        "distance": (("time", "latitude", "longitude"), distance_A),
    },
    coords={
        "time": cwv_orcestra.time,
        "latitude": cwv_orcestra.latitude,
        "longitude": cwv_orcestra.longitude,
    },
)
# %%
merged = xr.merge([cwv_orcestra, distance_ds])
lat_itcz_center = merged.distance.idxmin("latitude")

merged["distance_south"] = merged.distance.where(merged.latitude <= lat_itcz_center)
merged["distance_north"] = merged.distance.where(
    merged.latitude >= lat_itcz_center, drop=True
)

# %%

time_ind = 10

merged_ts = merged.isel(time=time_ind)

# merged_ts.tcwv.plot()
merged_ts.largest_cluster.plot.contour(levels=[0.5])
merged_ts.distance_north.plot(vmin=-20, vmax=20, cmap="seismic")

plt.scatter(merged_ts.longitude, lat_itcz_center.isel(time=time_ind))

# %%

for field in ["distance_south", "distance_north"]:
    test = merged_ts.groupby_bins(merged_ts[field], bins=np.arange(-5, 10, 1)).mean()
    test.tcwv.plot(label=field)
plt.legend()
# %%


# %%
