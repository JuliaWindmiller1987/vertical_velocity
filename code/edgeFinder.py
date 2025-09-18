import numpy as np
import scipy.interpolate
import scipy.signal
from sklearn.cluster import DBSCAN


def find_crossing(cwv, lat, cwv_thresh, sign_change=None):

    # sign_change: 2 if cwv increases with latitude
    # sign_change: -2 if cwv decreases with latitude

    cwv = cwv[np.argsort(lat)]
    lat = lat[np.argsort(lat)]

    cwv_shifted = cwv - cwv_thresh
    sign = np.sign(cwv_shifted)
    sign_diff = np.diff(sign)

    if np.all(sign_diff == 0):
        return np.nan

    if sign_change is None:
        sign_change_idx = np.where(sign_diff != 0)[0]
    else:
        sign_change_idx = np.where(sign_diff == sign_change)[0]

    return lat[sign_change_idx]


def closest_to_point(x, x_point):

    ind = np.argmin(np.abs(x - x_point))
    return x[ind]


def find_edges_numpy(
    cwv, lat_cwv_max, lat_cwv_south, lat_cwv_north, latitudes, cwv_thresh=45
):
    """
    Detect northern and southern moist margin as defined by CWV = cwv_thresh.
    To distinguish between the northern and southern moist margin,
    first determine the latitude of maximum CWV, larger or equal than cwv_thresh.

    Parameters:
        cwv (np.ndarray): 1D array of CWV values.
        lat_cwv_max (float): Latitude of CWV peak in the mean profile.
        latitudes (np.ndarray): 1D array of latitudes (same length as cwv).
        cwv_thresh (float): CWV of moist margin.
    """

    mask_nan = ~np.isnan(cwv)
    cwv = cwv[mask_nan]
    latitudes = latitudes[mask_nan]

    if np.nanmax(cwv) <= cwv_thresh:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peaks_i, _ = scipy.signal.find_peaks(cwv, height=cwv_thresh, prominence=2)
    if len(peaks_i) == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peak_lats = latitudes[peaks_i]

    lat_peak = closest_to_point(peak_lats, lat_cwv_max)

    north_mask = latitudes >= lat_peak
    north_cwv = cwv[north_mask]
    north_lats = latitudes[north_mask]

    lat_north_crossings = find_crossing(north_cwv, north_lats, cwv_thresh, -2)

    if np.all(np.isnan(lat_north_crossings)):
        lat_north = np.nan
    else:
        lat_north = closest_to_point(lat_north_crossings, lat_cwv_north)

    south_mask = latitudes <= lat_peak
    south_cwv = cwv[south_mask]
    south_lats = latitudes[south_mask]

    lat_south_crossings = find_crossing(south_cwv, south_lats, cwv_thresh, 2)

    if np.all(np.isnan(lat_south_crossings)):
        lat_south = np.nan
    else:
        lat_south = closest_to_point(lat_south_crossings, lat_cwv_south)

    return np.array([lat_south, lat_north], dtype=np.float32)


def find_cwv_center(cwv, lat_cwv_max, latitudes, cwv_thresh=45):

    mask_nan = ~np.isnan(cwv)
    cwv = cwv[mask_nan]
    latitudes = latitudes[mask_nan]

    peaks_i, _ = scipy.signal.find_peaks(cwv, height=cwv_thresh, prominence=2)
    if len(peaks_i) == 0:
        return np.array([np.nan], dtype=np.float32)

    peak_lats = latitudes[peaks_i]
    closest_peak_idx = np.argmin(np.abs(peak_lats - lat_cwv_max))
    lat_peak = peak_lats[closest_peak_idx]

    return np.array([lat_peak], dtype=np.float32)


def find_edge_points(cwv, cwv_thresh, delta_cwv):
    return cwv.where((cwv >= cwv_thresh - delta_cwv) & (cwv <= cwv_thresh + delta_cwv))


def rm_outlier(edge):

    points = list(zip(edge.longitude, edge))
    db = DBSCAN(eps=1.0, min_samples=5).fit(points)
    labels = db.labels_

    isolated_point_mask = labels == -1
    lat_south_connected = edge[~isolated_point_mask]

    return lat_south_connected
