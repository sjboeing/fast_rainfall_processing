from plot_utils import *

# import plot_utils
# import numpy as np
import matplotlib as mpl
import sys
import datetime
import os
import warnings
import argparse
warnings.filterwarnings("ignore")
from multiprocessing import Pool

# Requires python2.7, and surprisingly large amount of memory
yorks_labels = ["Leeds", "Sheffield", "York", "Hull", "Doncaster", "Northallerton"]
nat_labels = [
    "Leeds",
    "Sheff.",
    "York",
    "Hull",
    "B'burn",
    "Kendal",
    "M'boro.",
    "N'castle",
]
yorks_latlabels = [53.8008, 53.3811, 53.958, 53.744, 53.523, 54.338]
yorks_lonlabels = [-1.5491, -1.4701, -1.032, -0.333, -1.133, -1.429]
nat_latlabels = [53.8008, 53.3811, 53.958, 53.744, 53.750, 54.327, 54.555, 55.002]
nat_lonlabels = [-1.5491, -1.4701, -1.032, -0.333, -2.530, -2.559, -1.295, -1.727]
yorks_domain = Plot_Domain(
    (-2.6, 0.2, 53.25, 54.57),
    labels=yorks_labels,
    latlabels=yorks_latlabels,
    lonlabels=yorks_lonlabels,
)
nat_domain = Plot_Domain(
    (-3.55, 0.2, 53.15, 55.73),
    labels=nat_labels,
    latlabels=nat_latlabels,
    lonlabels=nat_lonlabels,
)
plot_domain = nat_domain
tnorm = mpl.colors.Normalize(vmin=0.0, vmax=24.0)

def proc_rain(ffile):
    ccube = iris.load(ffile)[0]
    np.ma.masked_less(ccube.data, 0.01)
    return ccube


def accums_mask(ffile):
    ccube = iris.load(ffile)[0]
    marray = np.ma.masked_less(ccube.data, 2.5)
    return marray.mask


def perc_max_accums(fcst_str, date, perc=95, period=180, radius=30, evl=False):
    datadir = fcstdirs + fcst_str[:6] + "/" + fcst_str + "/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    fcst_date = fcst_date + datetime.timedelta(hours=4)
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)

    mya = alphater()
    if evl:
        num = 20
        figdir = args.user + "/output_plots/evaluation/" + date_str + "/"
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
        f = MyPlot(
            5,
            4,
            num,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + fcst_str + r"_{}_max_T{}_accum".format(perc, period),
            figdir,
            lbotcol=True,
            suptitle=True,
        )
    else:
        num = 19
        figdir = (
            args.user
            + "/output_plots/forecasts/"
            + fcst_str[0:6]
            + "/"
            + fcst_str
            + "/"
        )
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
        f = MyPlot(
            5,
            4,
            num,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + date_str + r"_{}_max_T{}_accum".format(perc, period),
            figdir,
            lbotcol=False,
            suptitle=True,
        )
    # f.process_topo()
    f.start_bg()
    f.finish_bg()
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_ctopo.pdf '%(num)+f.file_str+'ctopo.pdf')
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_clines.pdf '%(num)+f.file_str+'clines.pdf')
    f.setup_decorations()
    if evl:
        f.append_subtitle(0, mya.next() + r") radar OBS")
        f.add_blegend(
            "Rainfall (mm); Fcst: %02dZ %02d/%02d/%04d; Valid: %02d/%02d/%04d"
            % (
                fcst_date.hour,
                fcst_date.day,
                fcst_date.month,
                fcst_date.year,
                date.day,
                date.month,
                date.year,
            )
        )
    else:
        f.add_legend(
            "Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"
            % (
                fcst_date.hour,
                fcst_date.day,
                fcst_date.month,
                fcst_date.year,
                date.day,
                date.month,
                date.year,
            )
        )
    for ii in range(num - 19, num - 1):
        f.append_subtitle(ii, mya.next() + r") member " + str(ii))
    f.append_subtitle(num - 1, mya.next() + r") ensemble")
    f.add_suptitle(
        "%02d/%02d/%04d %02dpp forecast max accumulation in T%s"
        % (date.day, date.month, date.year, perc, period)
    )
    f.process_decorations()
    f.setup_plots()
    if evl:
        ffile = (
            radardir + date_str + "_00_rad_pp_r{}_min_{}_tot.nc".format(radius, period)
        )
        ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        f.append_image(0, ccube)
    ffile = datadir + date_str + "_mem_pp_r{}_min_{}_tot.nc".format(radius, period)
    ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
    for ii in range(0, 18):
        f.append_image(ii + (num - 19), ccube[ii])
    ffile = datadir + date_str + "_ens_pp_r{}_min_{}_tot.nc".format(radius, period)
    try:
        ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
    except:
        ccube = iris.load_cube(args.user + "/output_plots/plot_fill.nc")
    f.append_image(num - 1, ccube)
    f.process_figure()
    del f


def exact_max_accums(fcst_str, date, period=180, evl=False):
    datadir = fcstdirs + fcst_str[0:6] + "/" + fcst_str + "/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    fcst_date = fcst_date + datetime.timedelta(hours=4)
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)

    mya = alphater()
    if evl:
        num = 19
        figdir = args.user + "/output_plots/evaluation/" + date_str
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
        f = MyPlot(
            5,
            4,
            num,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + fcst_str + "_exact_max_T%s_accum" % (period),
            figdir,
            suptitle=True,
        )
    else:
        num = 18
        figdir = (
            args.user
            + "/output_plots/forecasts/"
            + fcst_str[0:6]
            + "/"
            + fcst_str
            + "/"
        )
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
        f = MyPlot(
            5,
            4,
            num,
            plot_domain,
            date_str,
            fcst_str,
            figdir + date_str + "_exact_max_T%s_accum" % (period),
            figdir,
            suptitle=True,
        )
    # f.process_topo()
    f.start_bg()
    f.finish_bg()
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_ctopo.pdf '%(num)+f.file_str+'ctopo.pdf')
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_clines.pdf '%(num)+f.file_str+'clines.pdf')
    f.setup_decorations()
    if evl:
        f.append_subtitle(0, mya.next() + r") radar OBS")
    f.add_legend(
        "Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"
        % (
            fcst_date.hour,
            fcst_date.day,
            fcst_date.month,
            fcst_date.year,
            date.day,
            date.month,
            date.year,
        )
    )
    f.add_suptitle(
        "%02d/%02d/%04d exact forecast max accumulation in T%s"
        % (date.day, date.month, date.year, period)
    )
    for ii in range(0, 18):
        f.append_subtitle(ii + (num - 18), mya.next() + r") member " + str(ii))
    f.process_decorations()
    f.setup_plots()
    if evl:
        ffile = radardir + date_str + "_00_rad_exact_min_{}.nc".format(period)
        ccube = proc_rain(ffile)[0]
        f.append_image(0, ccube)
    ffile = datadir + date_str + "_exact_min_{}.nc".format(period)
    ccube = proc_rain(ffile)
    for ii in range(0, 18):
        f.append_image(ii + (num - 18), ccube[ii])
    f.process_figure()
    del f


def exact_day_accums(fcst_str, date, period=180, evl=False):
    mya = alphater()
    datadir = fcstdirs + fcst_str[:6] + "/" + fcst_str + "/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    fcst_date = fcst_date + datetime.timedelta(hours=4)
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)

    mya = alphater()
    if evl:
        num = 19
        figdir = args.user + "/output_plots/evaluation/" + date_str + "/"
    else:
        num = 18
        figdir = (
            args.user
            + "/output_plots/forecasts/"
            + fcst_str[0:6]
            + "/"
            + fcst_str
            + "/"
        )
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
    f = MyPlot(
        5,
        4,
        num,
        plot_domain,
        date_str,
        fcst_str,
        figdir + "/" + date_str + "_exact_day_accum",
        figdir,
        suptitle=True,
    )
    # f.process_topo()
    f.start_bg()
    f.finish_bg()
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_ctopo.pdf '%(num)+f.file_str+'ctopo.pdf')
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_clines.pdf '%(num)+f.file_str+'clines.pdf')
    f.setup_decorations()
    f.add_legend(
        "Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"
        % (
            fcst_date.hour,
            fcst_date.day,
            fcst_date.month,
            fcst_date.year,
            date.day,
            date.month,
            date.year,
        )
    )
    f.add_suptitle(
        "%02d/%02d/%04d exact forecast day total accumulation"
        % (date.day, date.month, date.year)
    )
    if evl:
        f.append_subtitle(0, mya.next() + r") radar OBS")
    for ii in range(0, 18):
        f.append_subtitle(ii + (num - 18), mya.next() + r") member " + str(ii))
    f.process_decorations()
    f.setup_plots()
    if evl:
        ffile = radardir + date_str + "_exact_tot.nc"
        ccube = proc_rain(ffile)
        f.append_image(0, ccube[0])
    ffile = datadir + date_str + "_exact_tot.nc"
    ccube = proc_rain(ffile)
    for ii in range(0, 18):
        f.append_image(ii + (num - 18), ccube[ii])
    f.process_figure()
    del f


def exact_accum_time(fcst_str, date, period=180, evl=False):
    datadir = fcstdirs + fcst_str[:6] + "/" + fcst_str + "/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    fcst_date = fcst_date + datetime.timedelta(hours=4)
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)

    mya = alphater()
    ccmap = mpl.cm.get_cmap("RdYlBu")
    listed_ccmap = matplotlib.colors.ListedColormap([ccmap(n) for n in range(256)])
    if evl:
        num = 19
        figdir = args.user + "/output_plots/evaluation/" + date_str + "/"
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
    else:
        num = 18
        figdir = (
            args.user
            + "/output_plots/forecasts/"
            + fcst_str[0:6]
            + "/"
            + fcst_str
            + "/"
        )
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)

    f = MyPlot(
        5,
        4,
        num,
        plot_domain,
        date_str,
        fcst_str,
        figdir + date_str + "_exact_T{}_time".format(period),
        figdir,
        norm=tnorm,
        cmap=listed_ccmap,
        levels=[],
        suptitle=True,
    )
    # f.process_topo()
    f.start_bg()
    f.finish_bg()
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_ctopo.pdf '%(num)+f.file_str+'ctopo.pdf')
    # os.system('cp '+args.user+'/output_plots/background_tiles/5_4_%s_clines.pdf '%(num)+f.file_str+'clines.pdf')
    f.setup_decorations()
    if evl:
        f.append_subtitle(0, mya.next() + r") radar OBS")
    for ii in range(0, 18):
        f.append_subtitle(ii + (num - 18), mya.next() + r") member " + str(ii))
    f.add_legend(
        "Time (hr)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"
        % (
            fcst_date.hour,
            fcst_date.day,
            fcst_date.month,
            fcst_date.year,
            date.day,
            date.month,
            date.year,
        )
    )
    f.add_suptitle(
        "%02d/%02d/%04d exact forecast time of T%s max accum periods > 2.5mm"
        % (date.day, date.month, date.year, period)
    )
    f.process_decorations()
    f.setup_plots()
    if evl:
        tfile = radardir + date_str + "_exact_min_{}_ind.nc".format(period)
        rfile = radardir + date_str + "_exact_min_{}.nc".format(period)
        ccube = iris.load_cube(tfile)
        mask = accums_mask(rfile)
        ccube.data = np.ma.masked_array(ccube.data, mask)
        ccube.data = ccube.data / 12.0 + 3.0
        f.append_image(0, ccube[0])
    tfile = datadir + date_str + "_exact_min_{}_ind.nc".format(period)
    rfile = datadir + date_str + "_exact_min_{}.nc".format(period)
    ccube = iris.load_cube(tfile)
    mask = accums_mask(rfile)
    ccube.data = np.ma.masked_array(ccube.data, mask)
    ccube.data = ccube.data / 12.0 + 3.0
    for ii in range(0, 18):
        f.append_image(ii + (num - 18), ccube[ii])
    f.process_figure()

    return f


def plot_forecasts(date, radius=30):
    exact_max_accums(fcst_str, date)
    exact_day_accums(fcst_str, date)
    exact_accum_time(fcst_str, date)
    for perc in [95, 98]:
        perc_max_accums(fcst_str, date, perc=perc)

parser = argparse.ArgumentParser()
parser.add_argument("--date_info", "-f", type=str, required=True)
parser.add_argument("--init", "-i", default=14, type=int, required=False)
parser.add_argument("--evl", "-e", required=False)
parser.add_argument(
    "--location",
    "-loc",
    type=str,
    required=False,
    default="/gws/nopw/j04/icasp_swf/bmaybee",
)
parser.add_argument(
    "--user", "-u", type=str, required=False, default="/home/users/bmaybee"
)
args = parser.parse_args()
fcstdirs = args.location + "/processed_forecasts/"

if args.evl is not None:
    date_str = args.date_info
    radardir = args.location + "/radar_obs/processed_radar/" + date_str + "_00/"
    # CHOOSE PREFERRED FORECAST TIME AND NUMBER OF INTERVALS
    date = datetime.datetime.strptime(date_str + "_14", "%Y%m%d_%H")
    fcst_strs, fcst_dates = [], []
    i = 48
    while i > 0:
        fcst_date = date - datetime.timedelta(hours=i)
        fcst_strs.append(
            "%04d%02d%02d_%02d"
            % (fcst_date.year, fcst_date.month, fcst_date.day, fcst_date.hour)
        )
        fcst_dates.append(fcst_date + datetime.timedelta(hours=4))
        i -= 12

    for fcst_str in fcst_strs:
        exact_max_accums(fcst_str, date, evl=True)
        for perc in [95, 98]:
            perc_max_accums(fcst_str, date, evl=True)

else:
    fcst_str = args.date_info + "_%02d" % args.init
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    days_ahead = []
    if fcst_date.year < 2019:
        out = 2
    else:
        out = 5

    for i in range(1, out):
        days_ahead.append(fcst_date + datetime.timedelta(days=i))

    nprocs = 16
    if __name__ == "__main__":
        p = Pool(min(len(days_ahead), nprocs))
        f = p.map(plot_forecasts, days_ahead)
