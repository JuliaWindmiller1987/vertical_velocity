import numpy as np
import scipy.signal


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
