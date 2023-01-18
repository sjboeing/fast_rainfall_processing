from plot_utils import *

# import plot_utils
# import numpy as np
import matplotlib as mpl
import sys
import datetime
import os
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# Requires python2.7, and surprisingly large amount of memory
def plot_forecasts(date, radius=30):
    labels = ["Leeds", "Sheffield", "York"]
    latlabels = [53.8008, 53.3811, 53.958]
    lonlabels = [-1.5491, -1.4701, -1.032]
    plot_domain = Plot_Domain(
        (-3.0, 0.2, 53.0, 54.8), labels=labels, latlabels=latlabels, lonlabels=lonlabels
    )
    zoom_domain = Plot_Domain(
        (-1.9, -0.8, 53.4, 54.3),
        labels=labels,
        latlabels=latlabels,
        lonlabels=lonlabels,
    )
    tnorm = mpl.colors.Normalize(vmin=0.0, vmax=24.0)

    def proc_rain(ffile):
        ccube = iris.load(ffile)[0]
        np.ma.masked_less(ccube.data, 0.01)
        return ccube

    def accums_mask(ffile):
        ccube = iris.load(ffile)[0]
        marray = np.ma.masked_less(ccube.data, 2.5)
        return marray.mask

    # datadir = "/gws/nopw/j04/icasp_swf/bmaybee/processed_forecasts/"+fcst_str[0:6]+"/"+fcst_str+"/"
    datadir = "manual_forecast_scripts/fast_rainfall_processing_files/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)
    fcst_date = fcst_date + datetime.timedelta(hours=4)

    # figdir = "/home/users/bmaybee/output_plots/forecasts/"+fcst_str[0:6]
    figdir = "/home/users/bmaybee/output_plots/sample_plots/"  # +fcst_str[0:6]
    if os.path.isdir(figdir) == False:
        os.system("mkdir " + figdir)
    figdir = figdir + "/" + fcst_str
    if os.path.isdir(figdir) == False:
        os.system("mkdir " + figdir)

    def perc_max_accums(perc, period):
        mya = alphater()
        f = MyPlot(
            5,
            4,
            19,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + date_str + "_{}_max_T{}_accum".format(perc, period),
            figdir,
            suptitle=True,
        )
        # f.process_topo()
        # f.start_bg()
        # f.finish_bg()
        os.system(
            "cp output_plots/background_tiles/5_4_19_ctopo.pdf "
            + f.file_str
            + "ctopo.pdf"
        )
        os.system(
            "cp output_plots/background_tiles/5_4_19_clines.pdf "
            + f.file_str
            + "clines.pdf"
        )
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
        for ii in range(0, 18):
            f.append_subtitle(ii, mya.next() + r") member " + str(ii))
        f.append_subtitle(18, mya.next() + r") ensemble")
        f.add_suptitle(
            "%02d/%02d/%04d %02dpp, r%r forecast max accumulation in T%s"
            % (date.day, date.month, date.year, perc, radius, period)
        )
        f.process_decorations()
        f.setup_plots()
        ffile = datadir + date_str + "_mem_pp_r{}_min_{}_tot.nc".format(radius, period)
        ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        for ii in range(0, 18):
            f.append_image(ii, ccube[ii])
        ffile = datadir + date_str + "_ens_pp_r{}_min_{}_tot.nc".format(radius, period)
        try:
            ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        except:
            ccube = iris.load_cube("/home/users/bmaybee/output_plots/plot_fill.nc")
        f.append_image(18, ccube)
        f.process_figure()
        del f

    for period in [60, 180]:
        for perc in [95, 98]:
            perc_max_accums(perc, period)

    def exact_max_accums(period):
        mya = alphater()
        f = MyPlot(
            5,
            4,
            18,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + date_str + "_exact_max_T{}_accum".format(period),
            figdir,
            suptitle=True,
        )
        # f.process_topo()
        # f.start_bg()
        # f.finish_bg()
        os.system(
            "cp output_plots/background_tiles/5_4_18_ctopo.pdf "
            + f.file_str
            + "ctopo.pdf"
        )
        os.system(
            "cp output_plots/background_tiles/5_4_18_clines.pdf "
            + f.file_str
            + "clines.pdf"
        )
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
            "%02d/%02d/%04d exact forecast max accumulation in T%s"
            % (date.day, date.month, date.year, period)
        )
        for ii in range(0, 18):
            f.append_subtitle(ii, mya.next() + r") member " + str(ii))
        f.process_decorations()
        f.setup_plots()
        ffile = datadir + date_str + "_exact_min_{}.nc".format(period)
        ccube = proc_rain(ffile)
        for ii in range(0, 18):
            f.append_image(ii, ccube[ii])
        f.process_figure()
        del f

    for period in [60, 180]:
        exact_max_accums(period)

    """
    def exact_day_accum(period=60)
        mya=alphater()
        f=MyPlot(5,4,18,plot_domain,date_str,fcst_str,figdir+'/'+date_str+'_exact_day_accum',figdir,suptitle=True)
        #f.process_topo()
        #f.start_bg()
        #f.finish_bg()
        os.system('cp output_plots/background_tiles/5_4_18_ctopo.pdf '+f.file_str+'ctopo.pdf')
        os.system('cp output_plots/background_tiles/5_4_18_clines.pdf '+f.file_str+'clines.pdf')
        f.setup_decorations()
        f.add_legend('Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d'%(fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year,date.day,date.month,date.year))
        f.add_suptitle('%02d/%02d/%04d exact forecast day total accumulation'%(date.day,date.month,date.year))
        for ii in range(0,18):
            f.append_subtitle(ii,mya.next()+r') member '+str(ii+1))
        f.process_decorations()
        f.setup_plots()
        ffile=datadir+date_str+'_m%02d_exact_tot.nc'%(ii)
        ccube=proc_rain(ffile)
        for ii in range(0,18):
            f.append_image(ii,ccube[ii])
        f.process_figure()
        del f
    #"""

    def exact_accum_time(period):
        mya = alphater()
        ccmap = mpl.cm.get_cmap("RdYlBu")
        listed_ccmap = matplotlib.colors.ListedColormap([ccmap(n) for n in range(256)])
        f = MyPlot(
            5,
            4,
            18,
            plot_domain,
            date_str,
            fcst_str,
            figdir + "/" + date_str + "_exact_T{}_time".format(period),
            figdir,
            norm=tnorm,
            cmap=listed_ccmap,
            levels=[],
            suptitle=True,
        )
        # f.process_topo()
        # f.start_bg()
        # f.finish_bg()
        os.system(
            "cp output_plots/background_tiles/5_4_18_ctopo.pdf "
            + f.file_str
            + "ctopo.pdf"
        )
        os.system(
            "cp output_plots/background_tiles/5_4_18_clines.pdf "
            + f.file_str
            + "clines.pdf"
        )
        f.setup_decorations()
        for ii in range(0, 18):
            f.append_subtitle(ii, mya.next() + r") member " + str(ii + 1))
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
        tfile = datadir + date_str + "_exact_min_{}_ind.nc".format(period)
        rfile = datadir + date_str + "_exact_min_{}.nc".format(period)
        ccube = iris.load_cube(tfile)
        mask = accums_mask(rfile)
        ccube.data = np.ma.masked_array(ccube.data, mask)
        ccube.data = ccube.data / 12.0 + 3.0
        for ii in range(0, 18):
            f.append_image(ii, ccube[ii])
        f.process_figure()
        del f

    exact_accum_time(60)


def upload_plots(fcst_str):
    netdir = (
        "/gws/nopw/j04/icasp_swf/public/forecast_plots/"
        + fcst_str[0:6]
        + "/"
        + fcst_str
    )
    if os.path.isdir(netdir) == False:
        os.system("mkdir " + netdir)
    os.system(
        "cp  /home/users/bmaybee/output_plots/forecasts/"
        + fcst_str[0:6]
        + "/"
        + fcst_str
        + "/* "
        + netdir
    )


fcst_str = str(sys.argv[1])
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
    p.map(plot_forecasts, days_ahead)

# upload_plots(fcst_str)
