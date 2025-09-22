# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import find_contours

# %%

data_path = "../data/"
file_name = "orcestra_era5_tcwv"
tcwv_cluster = xr.open_dataset(data_path + file_name + ".nc")

# %%

largest_cluster_pad = tcwv_cluster.largest_cluster
largest_cluster_pad[:, 0, :] = 0
largest_cluster_pad[:, -1, :] = 0

largest_cluster_pad = largest_cluster_pad.where(~np.isnan(largest_cluster_pad), 0)

res_in_deg = 0.25
cwv_thresh = 50

# %%


def contour_length(contours, res):
    cl = 0
    for c in range(len(contours)):
        delta_x = contours[c][:, 1][1:] - contours[c][:, 1][:-1]
        delta_y = contours[c][:, 0][1:] - contours[c][:, 0][:-1]
        cl += np.sum(np.sqrt(delta_x**2 + delta_y**2)) * res
    return cl


# %%

contour_lengths = np.empty(len(largest_cluster_pad.time))
contour_refs = np.empty(len(largest_cluster_pad.time))


for i_time in range(len(largest_cluster_pad.time)):
    largest_cluster_pad_ts = largest_cluster_pad.isel(time=i_time)

    i_contour_ds = largest_cluster_pad_ts

    i_contours = find_contours(i_contour_ds.values)
    contour_lengths[i_time] = contour_length(i_contours, res_in_deg)

    i_contour_lon = i_contour_ds.mean("latitude")
    contour_refs[i_time] = (
        i_contour_lon.where(i_contour_lon > 0, drop=True).count() * 2 * res_in_deg
    )


contour_ds = xr.Dataset(
    data_vars=dict(
        contour_lengths=(["time"], contour_lengths),
        contour_refs=(["time"], contour_refs),
    ),
    coords=dict(time=largest_cluster_pad.time),
)

contour_ds["blw"] = contour_ds["contour_refs"] / contour_ds["contour_lengths"]

# %%

# tcwv_cluster = xr.merge([tcwv_cluster, contour_ds])
# tcwv_cluster.to_netcdf(data_path + file_name + "_blw" + ".nc")

# %%

contour_ds["blw"].plot()

# %%
# Example
time = "2024-08-28T19:00:00.000000000"  # "2024-09-07T15"
largest_cluster_pad_ts = largest_cluster_pad.sel(time=time)

test_contour_ds = largest_cluster_pad_ts
test_contours = find_contours(test_contour_ds.values)

fig, ax = plt.subplots(1, 1)
plt.sca(ax)

plt.imshow(
    tcwv_cluster.tcwv.sel(time=time).values[::-1, :], origin="lower", cmap="Blues"
)

for c in range(len(test_contours)):

    plt.scatter(
        test_contours[c][:, 1], np.shape(tcwv_cluster.tcwv)[1] - test_contours[c][:, 0]
    )
    print(contour_length([test_contours[c]], res_in_deg))

print(contour_length(test_contours, res_in_deg))

# %%
