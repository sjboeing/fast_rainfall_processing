from plot_utils import *
import plot_utils
import numpy as np
import matplotlib as mpl
import datetime
import sys
import glob
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

##################################################################
# Extra script to generate forecast plots of RWCRS fields.
# NOTE: this script makes RWCRS radar plots only, not proxy flood maps. Script not necessary for generating FOREWARNS flood proxies.
#
# Argument parser options:
# -f : %Y%m%d string of radar observations date. Only compulsory argument
# -u : User directory (see documentation of requisite folder structure).
# -loc : Overarching data directory (see documentation of requisite folder structure).
#
# PREREQ's: integrated_rainfall_processing.py -r True
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--date", required=True, type=str)
parser.add_argument("-u", "--userdir", required=False, type=str, default='/home/users/bmaybee/')
parser.add_argument("-loc", "--location", required=False, type=str, default='/gws/nopw/j04/icasp_swf/bmaybee/')
args = parser.parse_args()
gws_root=args.location
user_root=args.userdir
if gws_root[-1] != "/":
    gws_root=gws_root+"/"
if user_root[-1] != "/":
    user_root=user_root+"/"

labels = ["Leeds", "Sheffield", "York"]
latlabels = [53.8008, 53.3811, 53.958]
lonlabels = [-1.5491, -1.4701, -1.032]
plot_domain = Plot_Domain((-5.8, 1.9, 49.9, 55.9), labels=[], latlabels=[], lonlabels=[])
tnorm = mpl.colors.Normalize(vmin=0.0, vmax=24.0)

def proc_rain(ffile):
    ccube = iris.load(ffile)[0]
    np.ma.masked_less(ccube.data, 0.01)
    return ccube
    
def accums_mask(ffile):
    ccube = iris.load(ffile)[0]
    marray = np.ma.masked_less(ccube.data, 2.5)
    return marray.mask

date_str=args.date
radardir=gws_root+"radar_obs/processed_radar/"+date_str+"_00/"
date = datetime.datetime.strptime(date_str, "%Y%m%d")

figdir = user_root + "output_plots/radar/"+date_str[0:6]
if os.path.isdir(figdir) == False:
    os.system("mkdir " + figdir)
"""
#Following script can be used to make images of hourly exact radar rainfall accumulations through day.
#Useful for understanding development/timing of events, but unwieldy on large domains.
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
    print ii, fois[ii]
    tstr=str(ttime.units.num2date(ttime.points)[fois[ii]])
    f.append_subtitle(ii,mya.next()+r') '+tstr[-8:])
f.add_blegend('Rainfall (mm)')
f.process_decorations()
f.setup_plots()
for ii in range(0,24):
    f.append_image(ii,(ccube[fois[ii]:fois[ii]+11].collapsed("time",iris.analysis.SUM)/12.))
f.process_figure()
del f
"""


def radar_accum_plots(radius):
    mya = alphater()
    f = MyPlot(4,2,7,plot_domain,date_str,"",figdir + "/" + date_str + "_radar_accum_plots",figdir,aheight=4.4,awidth=4.4,lbotcol=True,suptitle=True)
    # f.process_topo()
    f.start_bg()
    f.finish_bg()
    f.setup_decorations()
    f.append_subtitle(0, mya.next() + r") Exact max T60 accum")
    f.append_subtitle(1, mya.next() + r") Exact max T180 accum")
    f.append_subtitle(2, mya.next() + r") Exact max T360 accum")
    f.append_subtitle(3, mya.first() + r") Exact day accum")
    f.append_subtitle(4, mya.next() + r") 98pp. max T60 accum")
    f.append_subtitle(5, mya.next() + r") 98pp. max T180 accum")
    f.append_subtitle(6, mya.next() + r") 98pp. max T360 accum")
    f.add_legend("Rainfall (mm)")
    f.add_suptitle("%02d/%02d/%04d observed accumulations, exact and RWCRS" % (date.day, date.month, date.year))
    f.process_decorations()
    f.setup_plots()
    # ffile=radardir+date_str+'_00_rad_exact_tot.nc'
    # ccube=proc_rain(ffile)
    # ccube.data=ccube.data*32.
    # f.append_image(0,ccube)
    ffile = radardir + date_str + "_00_rad_exact_min_60.nc"
    ccube = proc_rain(ffile)
    f.append_image(0, ccube[0])
    ffile = radardir + date_str + "_00_rad_exact_min_180.nc"
    ccube = proc_rain(ffile)
    f.append_image(1, ccube[0])
    ffile = radardir + date_str + "_00_rad_exact_min_360.nc"
    ccube = proc_rain(ffile)
    f.append_image(2, ccube[0])
    ffile = radardir + date_str + "_00_rad_pp_r{}_min_60_tot.nc".format(radius)
    ccube = proc_rain(ffile)
    f.append_image(4, ccube.extract(iris.Constraint(percentile=98)))
    ffile = radardir + date_str + "_00_rad_pp_r{}_min_180_tot.nc".format(radius)
    ccube = proc_rain(ffile)
    f.append_image(5, ccube.extract(iris.Constraint(percentile=98)))
    ffile = radardir + date_str + "_00_rad_pp_r{}_min_360_tot.nc".format(radius)
    ccube = proc_rain(ffile)
    f.append_image(6, ccube.extract(iris.Constraint(percentile=98)))
    f.process_figure()
    del f


radar_accum_plots(30)
