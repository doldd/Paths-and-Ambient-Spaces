import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_subspace(x, y, landscape, aspect_equal=True, linear_color=False, interpolate=False, vmax=None, vmin=None, label='nll', ax=None):
    """
    :param x axis
    :param y axis
    :param landscape: loss surface
    :param aspect_equal: Draw x and y axis in same scale
    :param linear_color: if True use linear coloring else linear color steps according their quantiles
    :param df_samples: pandas dataframe of samples (must contain xx and yy columns for axis) or tuple with (dataframe, xname, yname)
    :param interpolate: if True use counterf to plot else pcolormesh
    :param vmax: max value of coloring
    :param vmin: min value of coloring
    """
    plt.rcParams['axes.grid'] = False
    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()
    cmap = plt.colormaps['viridis']
    if vmax is None:
        vmax = landscape.max()
    if vmin is None:
        vmin = landscape.min()
    if linear_color:
        norm = colors.Normalize(clip=True, vmax=vmax, vmin=vmin)
    else:
        # compute boundaries according data quantile (step_size=1/255)
        levels = [np.quantile(landscape[(landscape <= vmax) & (
            landscape >= vmin)], (1 / cmap.N * i)) for i in range(cmap.N)]
        levels.append(vmax)
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    if interpolate:
        if linear_color:
            pcm = ax.contourf(x, y, landscape, levels=cmap.N,
                              cmap=cmap, norm=norm)
        else:
            pcm = ax.contourf(x, y, landscape, levels=levels,
                              cmap=cmap, norm=norm)
    else:
        pcm = ax.pcolormesh(x, y, landscape, cmap=cmap,
                            norm=norm, shading='nearest', edgecolors=None, snap=True, linewidth=0, rasterized=True)
    if aspect_equal:
        ax.set_aspect('equal')
    if linear_color:
        fig.colorbar(pcm, ax=ax, label=label, shrink=0.5,
                     aspect=20*0.5, format="%1.2f")
    else:
        cb = fig.colorbar(pcm, ax=ax, label=label, shrink=0.5,
                          aspect=20*0.5, format="%1.2f", ticks=levels[::50])
        cb.ax.minorticks_off()
    # fig.legend()
    fig.tight_layout()
    # plt.legend()
    # plt.tight_layout()
    return ax

