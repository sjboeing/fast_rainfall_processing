## Jaiks, so much plotting stuff needed for this work
import matplotlib

matplotlib.use("agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as tiles
from skimage import exposure, restoration
import cartopy.feature as cfeature
import iris
import iris.plot as iplt
import matplotlib.pyplot as plt
import iris.analysis.cartography
from PIL import Image
from skimage.transform import rescale
from matplotlib.transforms import offset_copy
from matplotlib.animation import FuncAnimation
from iris.plot import _draw_2d_from_bounds
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl
from string import ascii_lowercase
from itertools import cycle
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import ListedColormap, to_rgba
import copy


class alphater:
    def __init__(self):
        self.value=cycle(ascii_lowercase)
    def next(self):
        return next(self.value)
    def first(self):
        self.value=cycle(ascii_lowercase)
        return next(self.value)


def paln(n_colors):
    sns.reset_orig()
    clrs = sns.color_palette("Paired", n_colors=n_colors)
    return clrs


def hlsn(n_colors):
    sns.reset_orig()
    clrs = sns.color_palette("hls", n_colors)
    return clrs


class MultiLineHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        legend_lines = []
        for line_index in range(12):
            legend_lines.append(
                plt.Line2D(
                    [
                        x0 + 0.37 * width * (line_index // 4),
                        x0 + (0.26 + 0.37 * (line_index // 4)) * width,
                    ],
                    [
                        0.33 * (line_index % 4) * height,
                        0.33 * (line_index % 4) * height,
                    ],
                    color=paln(12)[line_index],
                )
            )
        return legend_lines


############## Highly manual script to produce locations of axes
myweight = "black"
myfamily = "Liberation Sans"
fudgealpha = 0.8

coastlines = cfeature.NaturalEarthFeature(
    "physical", "coastline", "10m", edgecolor="black", facecolor="none"
)
states = cfeature.STATES.with_scale("10m")
crs_latlon = ccrs.PlateCarree()
# nws_precip_colors = [
# "#04e9e7",
# "#019ff4",
# "#0066db",
# "#02fd02",
# "#01c501",
# "#008e00",
# "#fdf802",
# "#e5bc00",
# "#fd9500",
# "#fd0000",
# "#d40000",
# "#bc0000",
# "#f800fd",
# "#9854c6",
# "#fdfdfd"
# ]
# precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
# levels = [0.01, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0,
# 25.0, 35.0, 50., 70.0, 100.0, 140.0, 200.0]

# tol_precip_colors = [
#    "#88ccee",
#    "#44aa99",
#    "#117733",
#    "#999933",
#    "#ddcc77",
#    "#cc6677",
#    "#882255",
#    "#aa4499",
#    "#332288",
#    "#000000"
# ]
# precip_colormap = matplotlib.colors.ListedColormap(tol_precip_colors)
# levels = [1.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]

# norm = matplotlib.colors.BoundaryNorm(levels, 10)

tol_precip_colors = [
    "#90C987",
    "#4EB256",
    "#7BAFDE",
    "#6195CF",
    "#F7CB45",
    "#EE8026",
    "#DC050C",
    "#A5170E",
    "#72190E",
    "#882E72",
    "#000000",
]

precip_colormap = matplotlib.colors.ListedColormap(tol_precip_colors)
precip_colormap.set_bad(alpha=0.0)
levels = [0.01, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0]

norm = matplotlib.colors.BoundaryNorm(levels, 12)

# Add scale bar
def scale_bar(ax, length=None, location=(0.7, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf
        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ["1", "2", "5"]:
                return int(x)
            else:
                return scale_number(x - 10**ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(
        bar_xs,
        [sby, sby],
        transform=tmc,
        color="k",
        alpha=fudgealpha,
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    # Plot the scalebar label
    ax.text(
        sbx,
        sby,
        str(length) + " km",
        transform=tmc,
        horizontalalignment="center",
        verticalalignment="bottom",
        weight=myweight,
        size=10.5,
        family=myfamily,
        alpha=fudgealpha,
    )


# Plotting classes
class Lightmap(tiles.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = "/gws/nopw/j04/icasp_swf/sboeing/tiles/toner/%s/%s/%s.png" % (z, x, y)
        return url

    def get_image(self, tile):
        url = self._image_url(tile)
        img = Image.open(url)
        img = img.convert(self.desired_tile_form)
        return img, self.tileextent(tile), "lower"


class Linesmap(tiles.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = "/gws/nopw/j04/icasp_swf/sboeing/tiles/tonerlines/%s/%s/%s.png" % (
            z,
            x,
            y,
        )
        return url

    def get_image(self, tile):
        url = self._image_url(tile)
        img = Image.open(url)
        img = img.convert(self.desired_tile_form)
        return img, self.tileextent(tile), "lower"


class Hillmap(tiles.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = "/gws/nopw/j04/icasp_swf/sboeing/tiles/hillshading/%s/%s/%s.png" % (
            z,
            x,
            y,
        )
        return url

    def get_image(self, tile):
        url = self._image_url(tile)
        img = Image.open(url)
        img = img.convert(self.desired_tile_form)
        return img, self.tileextent(tile), "lower"


class Plot_Domain:
    def __init__(self, extent, labels=None, latlabels=None, lonlabels=None):
        self.extent = extent
        self.labels = labels
        self.latlabels = latlabels
        self.lonlabels = lonlabels
        self.lightmap = Lightmap()
        self.linesmap = Linesmap()
        self.hillmap = Hillmap()

    def generate_maps(self, axis, level, fontsize=7, labels=False):
        self.generate_hillshademap(axis, level)
        self.generate_linesmap(axis, level, fontsize, labels)

    def generate_hillshademap(self, axis, level):
        axis.add_image(self.hillmap, level, zorder=0)

    def generate_linesmap(self, axis, level, fontsize, labels):
        axis.add_feature(coastlines, zorder=99)
        # axis.add_feature(states,zorder=99)
        #if labels:
        #    axis.add_image(self.lightmap, level, zorder=2)
        #else:
        #    axis.add_image(self.linesmap, level, zorder=2)
        geodetic_transform = ccrs.Geodetic()._as_mpl_transform(axis)
        text_transform = offset_copy(geodetic_transform, units="dots", x=-5)
        for label, latlabel, lonlabel in zip(
            self.labels, self.latlabels, self.lonlabels
        ):
            plt.plot(
                lonlabel,
                latlabel,
                marker="o",
                markersize=fontsize / 2 + 0.5,
                markerfacecolor="white",
                markeredgecolor="black",
                transform=crs_latlon,
                zorder=4,
            )
            plt.text(
                lonlabel,
                latlabel,
                label,
                verticalalignment="center",
                horizontalalignment="right",
                family=myfamily,
                transform=text_transform,
                weight=myweight,
                size=fontsize,
                alpha=fudgealpha,
            )
        scale_bar(axis, 60)

    def redraw_hillshademap(self, axis):
        for ii in [axis.get_images()[0]]:
            tempdata = ii._A[:]
            img = np.mean(tempdata[:, :, 0:3], axis=2) / 255.0
            escaled = exposure.equalize_adapthist(img, clip_limit=0.03)
            rimg = rescale(escaled, 2.0)
            self.newdata = np.dstack((rimg, rimg, rimg))
            ii.set_data(self.newdata)
            ii.set_rasterized(True)

    def redraw_linesmap(self, axis):
        for ii in [axis.get_images()[0]]:
            tempdata = ii._A[:]
            img = np.mean(tempdata[:, :, 0:3], axis=2) / 255.0
            rimg = rescale(img, 4.0)
            linesdata = np.dstack((rimg, rimg, rimg, 1.0 - 2.0 * rimg / 3.0))
            ii.set_data(linesdata)
            ii.set_rasterized(True)


class MyPlot:
    def __init__(
        self,
        nx,
        ny,
        nt,
        plot_domain,
        date_str,
        fcst_str,
        file_str,
        dir_str,
        dpi=100,
        dpi2=400,
        awidth=2.2,
        aheight=2.2,
        atop=0.3,
        aright=0.1,
        amarg=0.3,
        armarg=None,
        almarg=None,
        abotcol=0.5,
        norm=norm,
        cmap=precip_colormap,
        levels=levels,
        lbotcol=False,
        suptitle=False,
    ):
        self.dpi = dpi
        self.dpi2 = dpi2
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.ax = []
        self.fig = []
        self.plot_domain = plot_domain
        # self.file_str="/gws/nopw/j04/icasp_swf/public/test_plots/"+file_str
        self.date_str = date_str
        self.fcst_str = fcst_str
        self.file_str = file_str
        self.dir_str = dir_str
        self.nnow = 0
        self.awidth = awidth
        self.aheight = aheight
        self.atop = atop
        self.aright = aright
        self.amarg = amarg
        if almarg == None:
            self.almarg = self.amarg
        else:
            self.almarg = almarg
        if armarg == None:
            self.armarg = self.amarg
        else:
            self.armarg = armarg
        self.norm = norm
        self.cmap = cmap
        carrs = []
        for n in range(len(self.cmap.colors)):
            carr = np.array(to_rgba(self.cmap.colors[n])) * 0.5 + 0.4
            carr[-1] = 1.0
            carrs.append(tuple(carr))
        self.cmap2 = ListedColormap(carrs)
        self.levels = levels
        self.lbotcol = lbotcol
        self.suptitle = suptitle
        self.abotcol = abotcol
        self.size = self.my_size()

    # Size of entire plot
    def my_size(self):
        self.xtot = (
            self.nx * self.awidth + (self.nx - 1) * self.aright + 2 * self.amarg
        )  # Use width rather than totwidth for last image
        if self.lbotcol and self.suptitle:
            self.ytot = (
                self.ny * self.aheight
                + (self.ny - 1) * self.atop
                + 2 * self.amarg
                + 2 * self.abotcol
            )  # Extra margin at base to accommodate legend
        elif self.suptitle:
            self.ytot = (
                self.ny * self.aheight
                + (self.ny - 1) * self.atop
                + 2 * self.amarg
                + self.abotcol
            )  # Extra margin on top to accommodate title
        elif self.lbotcol:
            self.ytot = (
                self.ny * self.aheight
                + (self.ny - 1) * self.atop
                + 2 * self.amarg
                + self.abotcol
            )  # Extra margin on top to accommodate title
        else:
            self.ytot = (
                self.ny * self.aheight + (self.ny - 1) * self.atop + 2 * self.amarg
            )  # Extra margin on top to accommodate title
        return [self.xtot, self.ytot]

    # Location of a single axis
    def makeloc(self, axindex, factor=1.0):
        leftloc = axindex % self.nx
        bottomloc = self.ny - 1 - axindex // self.nx
        width = self.awidth * factor / self.xtot
        height = self.aheight / self.ytot
        totwidth = (self.awidth + self.aright) / self.xtot
        totheight = (self.aheight + self.atop) / self.ytot
        lmargin = self.amarg / self.xtot
        if self.lbotcol:
            bmargin = (self.amarg + self.abotcol) / self.ytot
        else:
            bmargin = self.amarg / self.ytot
        left = lmargin + leftloc * totwidth
        bottom = bmargin + bottomloc * totheight
        return [left, bottom, width, height]

    def makelocbot(self, factor=1.0, yfactor=1.0, yshift=0.0):
        width = (
            (self.awidth * self.nx + self.aright * (self.nx - 1)) * factor
        ) / self.xtot
        height = yfactor * self.abotcol / self.ytot
        lmargin = self.almarg / self.xtot
        bmargin = (
            self.amarg / self.ytot
            + (0.5 - 0.5 * yfactor) * self.abotcol / self.ytot
            + yshift / self.ytot
        )
        left = lmargin
        bottom = bmargin
        return [left, bottom, width, height]

    # Makes the topography maps
    def process_topo(self, title=""):
        self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
        self.ax = []
        for ii in range(self.nt):
            self.ax.append(
                plt.axes(self.makeloc(ii), projection=self.plot_domain.linesmap.crs)
            )
            self.ax[ii].set_extent(self.plot_domain.extent, crs=crs_latlon)
            self.ax[ii].set_rasterization_zorder(99)
            self.ax[ii].get_xaxis().set_visible(False)
            self.ax[ii].get_yaxis().set_visible(False)
            self.ax[ii].outline_patch.set_visible(False)
            self.plot_domain.generate_hillshademap(self.ax[ii], 8)
        self.topo_str = self.file_str + "topo.pdf"
        self.ctopo_str = self.file_str + "ctopo.pdf"
        plt.savefig(self.topo_str, dpi=self.dpi2)
        for ii in range(self.nt):
            self.plot_domain.redraw_hillshademap(self.ax[ii])
        plt.savefig(self.topo_str, dpi=self.dpi2)
        plt.close("all")
        os.system(
            "gs -DPDFSETTINGS=/prepress -dSAFER -dBATCH -dNOPAUSE -dColorImageFilter=/FlateEncode -dColorImageResolution="
            + str(self.dpi)
            + " -dMonoImageResolution="
            + str(self.dpi)
            + " -dGrayImageResolution="
            + str(self.dpi)
            + " -sDEVICE=pdfwrite -sOutputFile="
            + self.ctopo_str
            + '   -c "<< /GrayACSImageDict << /Blend 1 /VSamples [2 1 1 2] /QFactor 1.0 /HSamples [2 1 1 2] >> /ColorACSImageDict << /Blend 1 /VSamples [2 1 1 2] /QFactor 0.5 /HSamples [2 1 1 2] >> >> setdistillerparams " -f '
            + self.topo_str
        )

    def start_bg(self):
        self.fig = plt.figure(figsize=self.size, dpi=self.dpi2)
        self.ax = []
        for ii in range(self.nt):
            self.ax.append(
                plt.axes(self.makeloc(ii), projection=self.plot_domain.linesmap.crs)
            )
            self.ax[ii].set_extent(self.plot_domain.extent, crs=crs_latlon)
            self.ax[ii].background_patch.set_visible(False)
            self.plot_domain.generate_linesmap(self.ax[ii], 9, fontsize=7, labels=False)
        self.lines_str = self.file_str + "lines.pdf"
        self.clines_str = self.file_str + "clines.pdf"

    def add_legend(self, leg_string):
        tx_ax = plt.axes(self.makeloc(self.nt), frameon=False)
        tx_ax.get_xaxis().set_visible(False)
        tx_ax.get_yaxis().set_visible(False)
        t1 = plt.text(
            1.0,
            0.9,
            leg_string,
            verticalalignment="top",
            horizontalalignment="right",
            weight=myweight,
            family=myfamily,
            size=10,
            alpha=fudgealpha,
        )
        plt.text(
            1.0,
            0.1,
            "Toner map by Stamen \n Design, under CC BY 3.0. \n Data by OpenStreetMap, \n under ODbL.",
            verticalalignment="bottom",
            horizontalalignment="right",
            weight=myweight,
            family=myfamily,
            bbox=dict(facecolor="1.0", boxstyle="round"),
            size=6,
            alpha=fudgealpha,
        )
        cb_ax = plt.axes(self.makeloc(self.nt, factor=0.15))
        if self.levels == []:
            cb2 = matplotlib.colorbar.ColorbarBase(
                cb_ax, cmap=self.cmap, norm=self.norm
            )
        else:
            cb2 = matplotlib.colorbar.ColorbarBase(
                cb_ax,
                cmap=self.cmap,
                norm=self.norm,
                boundaries=self.levels,
                ticks=self.levels,
            )

    def add_blegend(self, leg_string, lnarrow=False):
        tx_ax = plt.axes(self.makelocbot(), frameon=False)
        tx_ax.get_xaxis().set_visible(False)
        tx_ax.get_yaxis().set_visible(False)
        if lnarrow:
            t1 = plt.text(
                0.0,
                0.0,
                leg_string,
                verticalalignment="bottom",
                horizontalalignment="left",
                weight=myweight,
                family=myfamily,
                size=10,
                alpha=fudgealpha,
            )
            # plt.text(1.0, 0.0, u'Toner map by Stamen Design, under CC BY 3.0. \n Data by OpenStreetMap, under ODbL.',
            #     verticalalignment='bottom', horizontalalignment='right',
            #     weight=myweight,family=myfamily,
            #     bbox=dict(facecolor='1.0', boxstyle='round'),size=8,alpha=fudgealpha)
            cb_ax = plt.axes(self.makelocbot(factor=1.0, yfactor=0.2, yshift=0.15))
        else:
            t1 = plt.text(
                0.58,
                0.2,
                leg_string,
                verticalalignment="bottom",
                horizontalalignment="left",
                weight=myweight,
                family=myfamily,
                size=10,
                alpha=fudgealpha,
            )
            # plt.text(1.0, 0.0, u'Toner map by Stamen \n Design, under CC BY 3.0. \n Data by OpenStreetMap, \n under ODbL.',
            #     verticalalignment='bottom', horizontalalignment='right',
            #     weight=myweight,family=myfamily,
            #     bbox=dict(facecolor='1.0', boxstyle='round'),size=8,alpha=fudgealpha)
            cb_ax = plt.axes(
                self.makelocbot(factor=0.55, yfactor=0.5),
            )
        if self.levels == []:
            cb2 = matplotlib.colorbar.ColorbarBase(
                cb_ax, cmap=self.cmap2, norm=self.norm, orientation="horizontal"
            )
        else:
            cb2 = matplotlib.colorbar.ColorbarBase(
                cb_ax,
                cmap=self.cmap2,
                norm=self.norm,
                boundaries=self.levels,
                ticks=self.levels,
                orientation="horizontal",
            )

    def add_suptitle(self, head_string):
        self.fig.suptitle(head_string, y=0.97, size=14, weight=myweight)

    def append_subtitle(self, ii, legend):
        self.ax[ii].set_title(legend, alpha=fudgealpha, fontsize=12, loc="left")

    def finish_bg(self):
        plt.savefig(self.lines_str, dpi=self.dpi2, transparent=True)
        #for ii in range(self.nt):
        #    self.plot_domain.redraw_linesmap(self.ax[ii])
        plt.savefig(self.lines_str, dpi=self.dpi2, transparent=True)
        plt.close("all")
        os.system(
            "gs -DPDFSETTINGS=/prepress -dSAFER -dBATCH -dNOPAUSE -dColorImageFilter=/FlateEncode -dColorImageResolution="
            + str(self.dpi)
            + " -dMonoImageResolution="
            + str(self.dpi)
            + " -dGrayImageResolution="
            + str(self.dpi)
            + " -sDEVICE=pdfwrite -sOutputFile="
            + self.clines_str
            + '   -c "<< /GrayACSImageDict << /Blend 1 /VSamples [2 1 1 2] /QFactor 1.0 /HSamples [2 1 1 2] >> /ColorACSImageDict << /Blend 1 /VSamples [2 1 1 2] /QFactor 0.5 /HSamples [2 1 1 2] >> >> setdistillerparams " -f '
            + self.lines_str
        )

    def setup_plots(self):
        self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
        self.ax = []
        self.im = []

    def append_image(self, ii, cube):
        self.ax.append(
            plt.axes(self.makeloc(ii), projection=self.plot_domain.linesmap.crs)
        )
        self.ax[-1].set_extent(self.plot_domain.extent, crs=crs_latlon)
        self.im.append(
            iplt.pcolormesh(
                cube, norm=self.norm, cmap=self.cmap, zorder=1, rasterized=True, alpha=0.8
            )
        )
        cube.data = np.ma.masked_less(cube.data, 0.01)
        self.im[-1].set_array(cube.data.ravel())

    def setup_decorations(self):
        self.fig = plt.figure(figsize=self.size, dpi=self.dpi)
        self.ax = []
        self.im = []
        for ii in range(self.nt):
            self.ax.append(
                plt.axes(self.makeloc(ii), projection=self.plot_domain.linesmap.crs)
            )
            self.ax[ii].set_extent(self.plot_domain.extent, crs=crs_latlon)

    def process_decorations(self):
        self.decor_str = self.file_str + "_decor.png"
        for ii in range(self.nt):
            try:
                self.ax[ii].get_xaxis().set_visible(False)
                self.ax[ii].get_yaxis().set_visible(False)
                self.ax[ii].outline_patch.set_visible(False)
                self.ax[ii].background_patch.set_visible(False)
            except:
                pass
        plt.savefig(self.decor_str, dpi=1.2 * self.dpi, transparent=True)
        plt.close("all")

    def process_figure(self):
        self.frame_str = self.file_str + ".png"
        self.tframe_str = self.file_str + "t.png"
        for ii in range(self.nt):
            try:
                self.ax[ii].get_xaxis().set_visible(False)
                self.ax[ii].get_yaxis().set_visible(False)
                self.ax[ii].add_feature(states, zorder=99, alpha=0.8)
                self.ax[ii].outline_patch.set_visible(False)
                self.ax[ii].background_patch.set_visible(False)
            except:
                pass
        plt.savefig(self.file_str, dpi=self.dpi, transparent=True)
        os.system(
            "convert "
            + self.frame_str
            + " -alpha on -channel A -evaluate set 90% +channel "
            + self.tframe_str
        )
        # os.system('optipng -quiet '+self.tframe_str)
        plt.close("all")
        print("Processing overlays")
        self.process_overlays()
        self.cleanup()

    def process_overlays(self):
        f = open(self.file_str + ".tex", "w")
        f.write(
            r"""\documentclass{article}
\usepackage[paperwidth="""
            + str(self.size[0])
            + r"""in,paperheight="""
            + str(self.size[1])
            + r"""in,top=0mm, bottom=0mm, left=0mm, right=0mm]{geometry}
\usepackage{graphicx}
\usepackage{fontenc}
\usepackage{overpic}
\begin{document}
\begin{figure}
\centering   
\begin{overpic}[width=\paperwidth]{"""
            + self.file_str
            + r"""t.png}
   \put(0,0){\includegraphics[width=\paperwidth]{"""
            + self.file_str
            + r"""_decor.png}}  
   \put(0,0){\includegraphics[width=\paperwidth]{"""
            + self.file_str
            + r"""clines.pdf}}  
\end{overpic}
\end{figure}
\clearpage
\end{document}
"""
        )
        f.close()

    def cleanup(self):
        os.system("pdflatex --output-directory="+ self.dir_str+ " "+ self.file_str+ ".tex > /dev/null 2>&1")
        #os.system("pdflatex --output-directory="+ self.dir_str+ " "+ self.file_str+ ".tex")
        plt.close("all")
        os.system("rm -f " + self.file_str + "c*")
        os.system("rm -f " + self.file_str + "lines*")
        os.system("rm -f " + self.file_str + "t*")
        os.system("rm -f " + self.file_str + "t*")
        os.system("rm -f " + self.file_str + ".aux")
        os.system("rm -f " + self.file_str + ".log")
        os.system("rm -f " + self.file_str + ".png")
        os.system("rm -f " + self.file_str + "_decor.png")
        os.system("rm -f " + self.file_str + ".tex")


def extract_hyetograph(ffile, lon, lat):
    rainfall = iris.load(ffile)[0]
    try:
        rot_pole = rainfall.coord("grid_latitude").coord_system.as_cartopy_crs()
        ll = ccrs.Geodetic()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [
            ("grid_latitude", target_xy[1]),
            ("grid_longitude", target_xy[0]),
        ]
    except:
        rot_pole = rainfall.coord(
            "projection_y_coordinate"
        ).coord_system.as_cartopy_crs()
        ll = ccrs.Geodetic()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [
            ("projection_y_coordinate", target_xy[1]),
            ("projection_x_coordinate", target_xy[0]),
        ]
    rainfall_series = rainfall.interpolate(sample_points, iris.analysis.Nearest())
    return rainfall_series.data


def extract_pdf(ffile, nfile, lon, lat):
    rainfall = iris.load(ffile)[0]
    nrainfall = iris.load(nfile)[0]
    try:
        rot_pole = rainfall.coord("grid_latitude").coord_system.as_cartopy_crs()
        ll = ccrs.Geodetic()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [
            ("grid_latitude", target_xy[1]),
            ("grid_longitude", target_xy[0]),
        ]
    except:
        rot_pole = rainfall.coord(
            "projection_y_coordinate"
        ).coord_system.as_cartopy_crs()
        ll = ccrs.Geodetic()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [
            ("projection_y_coordinate", target_xy[1]),
            ("projection_x_coordinate", target_xy[0]),
        ]
    rainfall_series = rainfall.interpolate(sample_points, iris.analysis.Nearest())
    nrainfall_out = nrainfall.interpolate(sample_points, iris.analysis.Nearest())
    return rainfall_series.data, nrainfall_out.data
