import numpy as np
import scipy.signal
from sklearn.cluster import DBSCAN


def find_edges_numpy(cwv, lat_cwv_max, latitudes, cwv_thresh=45):
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

    if np.max(cwv) <= cwv_thresh:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peaks_i, _ = scipy.signal.find_peaks(cwv, height=cwv_thresh, prominence=2)
    if len(peaks_i) == 0:
        return np.array([np.nan, np.nan], dtype=np.float32)

    peak_lats = latitudes[peaks_i]
    closest_peak_idx = np.argmin(np.abs(peak_lats - lat_cwv_max))
    lat_peak = peak_lats[closest_peak_idx]

    north_mask = latitudes >= lat_peak
    north_cwv = cwv[north_mask]
    north_lats = latitudes[north_mask]
    lat_north = (
        np.min(north_lats[north_cwv <= cwv_thresh])
        if np.any(north_cwv <= cwv_thresh)
        else np.nan
    )

    south_mask = latitudes <= lat_peak
    south_cwv = cwv[south_mask]
    south_lats = latitudes[south_mask]
    lat_south = (
        np.max(south_lats[south_cwv <= cwv_thresh])
        if np.any(south_cwv <= cwv_thresh)
        else np.nan
    )

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
