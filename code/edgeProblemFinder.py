# %%
import sys

sys.path.append("./code")
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.cluster import DBSCAN


import importlib
import edgeFinder

importlib.reload(edgeFinder)
from edgeFinder import find_edges_numpy, rm_outlier, find_crossing

from plotUtils import plot_cwv_field, add_east_west_boxes, plot_map

# %%

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)
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
from edgeFinder import find_edges_numpy, rm_outlier

era_ind = cwv_orcestra_with_edge.sel(
    time="2024-09-30T12:00:00.000000000", method="nearest"
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

    def rm_outlier(edge):

        points = list(zip(edge.longitude, edge))
        db = DBSCAN(eps=1.0, min_samples=5).fit(points)
        labels = db.labels_

        isolated_point_mask = labels == -1
        lat_south_connected = edge[~isolated_point_mask]

        return lat_south_connected

    edge_i = edges_i.dropna(dim="longitude")
    edge_connected_i = rm_outlier(edge_i)
    edge_connected_i.plot(c="k")

    s = 5

    spl = interpolate.make_splrep(edge_connected_i.longitude, edge_connected_i, s=s)
    lat_fit = spl(edge_connected_i.longitude)

    plt.scatter(edge_connected_i.longitude, lat_fit, label=s)
    edge_connected_i.plot(c="k")
# %%
