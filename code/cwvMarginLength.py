# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import find_contours

# %%

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)

file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc"
cwv_orcestra = xr.open_dataset(data_path + file_name)
time_da = cwv_orcestra.time

res_in_deg = 0.25
cwv_thresh = 48
min_contour_length = 6.0

# %%


def cluster_field_with_border(da, thresh):
    field_with_border = xr.where(da > thresh, 1, 0)
    field_with_border[:, 0, :] = 0
    field_with_border[:, -1, :] = 0
    return field_with_border


def contour_length(contours, res, min_contour_length):
    cl = 0
    for c in range(len(contours)):
        delta_x = contours[c][:, 1][1:] - contours[c][:, 1][:-1]
        delta_y = contours[c][:, 0][1:] - contours[c][:, 0][:-1]
        length_c = np.sum(np.sqrt(delta_x**2 + delta_y**2)) * res
        if length_c >= min_contour_length:
            cl += length_c
    return cl


def ref_length(binary_field):
    i_contour_lon = binary_field.mean("latitude")
    return i_contour_lon.where(i_contour_lon > 0, drop=True).count() * 2 * res_in_deg


def calc_blw_ds(cluster_fields, res_in_deg, min_contour_length):

    times_ds = cluster_fields.time
    for i_time in range(len(times_ds)):

        cluster_fields_ts = cluster_fields.isel(time=i_time)

        contour_lengths[i_time] = contour_length(
            find_contours(cluster_fields_ts.values, level=0.5),
            res_in_deg,
            min_contour_length,
        )

        contour_refs[i_time] = ref_length(cluster_fields_ts)

    contour_ds = xr.Dataset(
        data_vars=dict(
            contour_lengths=(["time"], contour_lengths),
            contour_refs=(["time"], contour_refs),
        ),
        coords=dict(time=times_ds),
    )

    contour_ds["blw"] = contour_ds["contour_refs"] / contour_ds["contour_lengths"]

    return contour_ds


# %%

contour_lengths = np.empty(len(time_da))
contour_refs = np.empty(len(time_da))

cluster_fields = cluster_field_with_border(cwv_orcestra.tcwv, cwv_thresh)
cluster_fields_sens = cluster_field_with_border(cwv_orcestra.tcwv, 45)

contour_ds = calc_blw_ds(cluster_fields, res_in_deg, min_contour_length)
contour_ds_sens = calc_blw_ds(cluster_fields_sens, res_in_deg, min_contour_length)

# %%

contour_ds.blw.plot()
contour_ds_sens.blw.plot()
# %%

# tcwv_cluster = xr.merge([tcwv_cluster, contour_ds])
# tcwv_cluster.to_netcdf(data_path + file_name + "_blw" + ".nc")

# %%

contour_ds["blw"].plot()

# %%
# Example
time = "2024-08-15T12:00:00.000000000"  # "2024-09-07T15"
cluster_fields_ts = cluster_fields.sel(time=time)

test_contour_ds = cluster_fields_ts
test_contours = find_contours(test_contour_ds.values, level=0.5)

fig, ax = plt.subplots(1, 1)
plt.sca(ax)

plt.imshow(
    cluster_fields.sel(time=time).values[::-1, :],
    origin="lower",
    cmap="Blues",
    vmin=46,
    vmax=64,
)

for c in range(len(test_contours)):

    c_length = contour_length([test_contours[c]], res_in_deg, min_contour_length)

    if c_length >= 3.0:

        plt.scatter(
            test_contours[c][:, 1],
            np.shape(cluster_fields)[1] - test_contours[c][:, 0],
        )
        print(c_length)

print(contour_length(test_contours, res_in_deg, min_contour_length))

# %%
