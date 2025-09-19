import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import seaborn as sb
import numpy as np

levels_cwv = np.sort(np.unique([46, 48, 50, 52, 54]))
extent = [-65, -15, -5, 25]


def plot_map():
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        sharex=True,
    )

    extent = [-65, -15, -5, 25]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=1.0)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25)
    ax.set_title(" ")

    add_east_west_boxes(ax)

    return fig, ax


def plot_cwv_field(cwv_mean, levels=levels_cwv, extent=extent, ax=None, **kwargs):
    """
    Plot the mean CWV field with optional contour levels.

    Parameters:
        cwv_mean (xr.DataArray): 2D CWV field with dimensions (latitude, longitude)
        levels (list or np.ndarray): Contour levels to overlay
        extent (list): [lon_min, lon_max, lat_min, lat_max] map extent
        ax (GeoAxesSubplot): Optional matplotlib axes to plot into

    Returns:
        ax: The axis used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )

    # Set extent and background features
    if extent is None:
        extent = [
            cwv_mean.longitude.min(),
            cwv_mean.longitude.max(),
            cwv_mean.latitude.min(),
            cwv_mean.latitude.max(),
        ]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(alpha=1.0)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.25)

    plot_kwargs = {**dict(alpha=0.75, cmap="Blues", vmin=45, vmax=70), **kwargs}

    # Main CWV field
    cwv_mean.plot(ax=ax, **plot_kwargs)

    # Optional contours
    if levels is not None:
        cwv_mean.plot.contour(ax=ax, levels=levels, colors="k", linewidths=1, alpha=0.5)

    sb.despine()
    return ax


def add_east_west_boxes(
    ax,
    lat_min=1,
    lat_max=22,
    lon_east_min=-40,
    lon_east_max=-19,
    lon_west_min=-61,
    lon_west_max=-40,
    col_east="C0",
    col_west="C1",
):
    """
    Add East and West boxes to a Cartopy axis.

    Parameters:
        ax (GeoAxesSubplot): Matplotlib axis with Cartopy projection
        lat_min, lat_max (float): Latitude bounds for both boxes
        lon_east_min, lon_east_max (float): Longitude bounds for East box
        lon_west_min, lon_west_max (float): Longitude bounds for West box
        col_east, col_west (str): Colors for the box edges and labels

    Returns:
        None (modifies ax in place)
    """

    east_box_coords = [
        [lon_east_min, lat_min],
        [lon_east_max, lat_min],
        [lon_east_max, lat_max],
        [lon_east_min, lat_max],
    ]

    west_box_coords = [
        [lon_west_min, lat_min],
        [lon_west_max, lat_min],
        [lon_west_max, lat_max],
        [lon_west_min, lat_max],
    ]

    # Create and add East box
    east_box = mpatches.Polygon(
        east_box_coords,
        closed=True,
        edgecolor=col_east,
        facecolor="none",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(east_box)

    # Create and add West box
    west_box = mpatches.Polygon(
        west_box_coords,
        closed=True,
        edgecolor=col_west,
        facecolor="none",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(west_box)

    # Add text labels
    ax.text(
        (lon_east_min + lon_east_max) / 2,
        lat_max + 1,
        "East Atlantic",
        color=col_east,
        transform=ccrs.PlateCarree(),
        ha="center",
    )

    ax.text(
        (lon_west_min + lon_west_max) / 2,
        lat_max + 1,
        "West Atlantic",
        color=col_west,
        transform=ccrs.PlateCarree(),
        ha="center",
    )

    sb.despine()
