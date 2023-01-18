from plot_utils import *
import plot_utils
import numpy as np
import matplotlib as mpl
import datetime
import sys
import glob
import os
import warnings
import argparse

warnings.filterwarnings("ignore")

labels = ["Leeds", "Sheffield", "York"]
latlabels = [53.8008, 53.3811, 53.958]
lonlabels = [-1.5491, -1.4701, -1.032]
plot_domain = Plot_Domain(
    (-3.0, 0.2, 53.0, 54.8), labels=labels, latlabels=latlabels, lonlabels=lonlabels
)
zoom_domain = Plot_Domain(
    (-1.9, -1.4, 53.6, 54.0), labels=labels, latlabels=latlabels, lonlabels=lonlabels
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


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--date", required=True, type=str)
parser.add_argument(
    "-loc",
    "--location",
    default="/gws/nopw/j04/icasp_swf/bmaybee",
    required=False,
    type=str,
)
parser.add_argument(
    "-u", "--user", default="/home/users/bmaybee", required=False, type=str
)
args = parser.parse_args()
gws_root = args.location
user_root = args.user

date_str = args.date
radardir = gws_root + "/radar_obs/processed_radar/" + date_str + "_00/"
# radardir = "/home/users/bmaybee/manual_forecast_scripts/fast_rainfall_processing_files/"+date_str+"_00/"
date = datetime.datetime.strptime(date_str, "%Y%m%d")

figdir = user_root + "/output_plots/radar/" + date_str[0:6]
# figdir = "/home/users/bmaybee/output_plots/sample_plots/radar_tests"
if os.path.isdir(figdir) == False:
    os.system("mkdir " + figdir)
"""
#RADAR
mya=alphater()
ffile=glob.glob(radardir+"/../../%04d/*"%(date.year)+date_str+"*.nc")
#ffile=radardir+date_str+'_00_rad_exact.nc'
ccube=proc_rain(ffile)
fois=range(0,24*12,12)
f=MyPlot(6,4,24,plot_domain,date_str,'',figdir+'/'+date_str+'_radar_accums',figdir,lbotcol=True,suptitle=True)
#f.process_topo()
#f.start_bg()
#f.finish_bg()
os.system('cp output_plots/background_tiles/6_4_24_ctopo.pdf '+f.file_str+'ctopo.pdf')
os.system('cp output_plots/background_tiles/6_4_24_clines.pdf '+f.file_str+'clines.pdf')
ttime=ccube.coord('time')
tstr=str(ttime.units.num2date(ttime.points)[fois[0]])
f.setup_decorations()
f.append_subtitle(0,mya.first()+r') '+tstr[-8:])
f.add_suptitle('Hourly accumulations, %02d/%02d/%04d'%(date.day,date.month,date.year))
for ii in range(1,24):
    #print(ii, fois[ii])
    tstr=str(ttime.units.num2date(ttime.points)[fois[ii]])
    f.append_subtitle(ii,mya.next()+r') '+tstr[-8:])
f.add_blegend('Rainfall (mm)')
f.process_decorations()
f.setup_plots()
for ii in range(0,24):
    f.append_image(ii,(ccube[fois[ii]:fois[ii]+11].collapsed("time",iris.analysis.SUM)/12.))
f.process_figure()
del f
#"""


def radar_accum_plots(radius):
    mya = alphater()
    f = MyPlot(
        5,
        1,
        5,
        plot_domain,
        date_str,
        "",
        figdir + "/" + date_str + "_radar_accum_plots",
        figdir,
        lbotcol=True,
        suptitle=True,
    )
    # f.process_topo()
    # f.start_bg()
    # f.finish_bg()
    os.system(
        "cp output_plots/background_tiles/5_1_5_ctopo.pdf " + f.file_str + "ctopo.pdf"
    )
    os.system(
        "cp output_plots/background_tiles/5_1_5_clines.pdf " + f.file_str + "clines.pdf"
    )
    f.setup_decorations()
    f.append_subtitle(0, mya.first() + r") Exact day accum")
    f.append_subtitle(1, mya.next() + r") Exact max T60 accum")
    f.append_subtitle(3, mya.next() + r") Exact max T180 accum")
    f.append_subtitle(2, mya.next() + r") 95pp. max T180 accum")
    f.append_subtitle(4, mya.next() + r") 98pp. max T180 accum")
    f.add_blegend("Rainfall (mm)")
    f.add_suptitle(
        "%02d/%02d/%04d observed max 1H, 3H and day total accumulations"
        % (date.day, date.month, date.year)
    )
    f.process_decorations()
    f.setup_plots()
    ffile = radardir + date_str + "_00_rad_exact_tot.nc"
    ccube = proc_rain(ffile)
    f.append_image(0, ccube[0])
    ffile = radardir + date_str + "_00_rad_exact_min_60.nc"
    ccube = proc_rain(ffile)
    f.append_image(1, ccube[0])
    ffile = radardir + date_str + "_00_rad_exact_min_180.nc"
    ccube = proc_rain(ffile)
    f.append_image(2, ccube[0])
    ffile = radardir + date_str + "_00_rad_pp_r{}_min_180_tot.nc".format(radius)
    ccube = proc_rain(ffile)
    f.append_image(3, ccube.extract(iris.Constraint(percentile=95)))
    f.append_image(4, ccube.extract(iris.Constraint(percentile=98)))
    f.process_figure()
    del f


radar_accum_plots(30)


def radar_time_plots(run):
    mya = alphater()
    ccmap = mpl.cm.get_cmap("RdYlBu")
    listed_ccmap = matplotlib.colors.ListedColormap([ccmap(n) for n in range(256)])
    f = MyPlot(
        3,
        1,
        2,
        plot_domain,
        date_str,
        "",
        figdir + "/" + date_str + "_radar_time_plots",
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
        "cp output_plots/background_tiles/3_1_2_ctopo.pdf " + f.file_str + "ctopo.pdf"
    )
    os.system(
        "cp output_plots/background_tiles/3_1_2_clines.pdf " + f.file_str + "clines.pdf"
    )
    f.setup_decorations()
    f.append_subtitle(0, mya.next() + r") Exact max 1hr accum")
    f.append_subtitle(1, mya.next() + r") Exact max 3hr accum")
    f.add_suptitle(
        "%02d/%02d/%04d obs time of max accum periods > 2.5mm"
        % (date.day, date.month, date.year)
    )
    f.add_legend("Time (hr)")
    f.process_decorations()
    f.setup_plots()
    tfile = radardir + date_str + "_00_rad_exact_min_60_ind.nc"
    rfile = radardir + date_str + "_00_rad_exact_min_60.nc"
    ccube = iris.load_cube(tfile)
    mask = accums_mask(rfile)
    ccube.data = np.ma.masked_array(ccube.data, mask)
    ccube.data = ccube.data / 12.0 + 3.0
    f.append_image(0, ccube[0])
    tfile = radardir + date_str + "_00_rad_exact_min_180_ind.nc"
    rfile = radardir + date_str + "_00_rad_exact_min_180.nc"
    ccube = iris.load_cube(tfile)
    mask = accums_mask(rfile)
    ccube.data = np.ma.masked_array(ccube.data, mask)
    ccube.data = ccube.data / 12.0 + 3.0
    f.append_image(1, ccube[0])
    f.process_figure()
    del f


radar_time_plots(True)

print("Plotting done, putting online")
# os.system('rm /home/users/bmaybee/output_plots/radar/'+date_str[0:6]+'/'+date_str+'*lines.pdf')
# os.system('rm /home/users/bmaybee/output_plots/radar/'+date_str[0:6]+'/'+date_str+'*topo.pdf')
# os.system('cp /home/users/bmaybee/output_plots/radar/'+date_str[0:6]+'/'+date_str+'* /gws/nopw/j04/icasp_swf/public/radar_plots/'+date_str[0:6])
# os.system('. /home/users/bmaybee/testbed_obs/make_obs_html.sh '+date_str)
