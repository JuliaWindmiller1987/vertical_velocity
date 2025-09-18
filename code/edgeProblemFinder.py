# %%
import sys

sys.path.append("./code")
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.ndimage import label, binary_erosion, binary_dilation, distance_transform_edt

# %%

data_path = (
    "/Users/juliawindmiller/MPI/Windmiller2025_ObservingVerticalVelocities/data/"
)

file_name = "msk_tcwv-2024-08-09-1hr_22_1_-61_-19.nc"
cwv_orcestra = xr.open_dataset(data_path + file_name)


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
cwv_orcestra_with_edge = xr.merge([cwv_orcestra, distance_ds])
cwv_orcestra_with_edge = cwv_orcestra_with_edge.sel(longitude=slice(300, 340))
cwv_orcestra_with_edge = cwv_orcestra_with_edge.sel(latitude=slice(21, 2))

lat_itcz_center = cwv_orcestra_with_edge.distance.idxmin("latitude")

cwv_orcestra_with_edge["distance_south"] = cwv_orcestra_with_edge.distance.where(
    cwv_orcestra_with_edge.latitude <= lat_itcz_center
)
cwv_orcestra_with_edge["distance_north"] = cwv_orcestra_with_edge.distance.where(
    cwv_orcestra_with_edge.latitude >= lat_itcz_center, drop=True
)

cwv_orcestra.to_netcdf(data_path + file_name.split(".")[0] + "_w_edge_field" + ".nc")
# %%
# Example use case

time_ind = 10
cwv_orcestra_with_edge_ts = cwv_orcestra_with_edge.isel(time=time_ind)

# cwv_orcestra_with_edge_ts.tcwv.plot()
cwv_orcestra_with_edge_ts.largest_cluster.plot.contour(levels=[0.5])
cwv_orcestra_with_edge_ts.distance_north.plot(vmin=-20, vmax=20, cmap="seismic")

plt.scatter(cwv_orcestra_with_edge_ts.longitude, lat_itcz_center.isel(time=time_ind))

# %%

dic_region = {"west": [300, 320], "east": [320, 340], "all": [300, 340]}
linestyles = ["solid", "dashed"]

for i_region, region in enumerate(["all"]):
    lon_min, lon_max = dic_region[region]
    cwv_orcestra_with_edge_ts_region = cwv_orcestra_with_edge_ts.sel(
        longitude=slice(lon_min, lon_max)
    )
    for i_field, field in enumerate(["distance_south", "distance_north"]):
        test = cwv_orcestra_with_edge_ts_region.groupby_bins(
            cwv_orcestra_with_edge_ts_region[field], bins=np.arange(-5, 10, 0.5)
        ).mean()
        test.tcwv.plot(
            label=f"{field} ({region})", c=f"C{i_field}", linestyle=linestyles[i_region]
        )
    plt.legend()

plt.axvline(0, c="k", linestyle="dashed")
sb.despine()
# %%
