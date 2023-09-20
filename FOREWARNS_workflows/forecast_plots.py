from plot_utils import *

# import plot_utils
# import numpy as np
import matplotlib as mpl
import sys
import datetime
import os
import warnings
import argparse
from multiprocessing import Pool
warnings.filterwarnings("ignore")

##################################################################
# Extra script to generate forecast plots of RWCRS fields.
# NOTE: this script makes RWCRS rainfall forecast plots only, not flood forecasts. Script not necessary for generating FOREWARNS flood forecasts.
#
# Argument parser options:
# -f : %Y%m%d string of forecast initialisation date. Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK)
# -d : If only want plots for single calendar day, use this option. Takes %Y%m%d string of forecast validity date.
# -e : Default setting is to plot forecast fields only. Set true to make evaluation plots which include radar OBS stamp; requires -d option if used.
# -u : User directory (see documentation of requisite folder structure).
# -loc : Overarching data directory (see documentation of requisite folder structure).
#
# PREREQ's: integrated_rainfall_processing.py
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fcst_date", required=True, type=str)
parser.add_argument("-i", "--fcst_init", required=False, default=0, type=str)
parser.add_argument("-d", "--date", required=False, type=str)
parser.add_argument("-e", "--eval", required=False, default=False, type=bool)
parser.add_argument("-u", "--userdir", required=False, default="/home/users/bmaybee/", type=str)
parser.add_argument("-loc", "--location", required=False, default="/gws/nopw/j04/icasp_swf/bmaybee/", type=str)
args = parser.parse_args()
gws_root=args.location
user_root=args.userdir
if gws_root[-1] != "/":
    gws_root=gws_root+"/"
if user_root[-1] != "/":
    user_root=user_root+"/"
fcst_str=args.fcst_date+"_"+(args.fcst_init).zfill(2)
print(fcst_str)

def plot_forecasts(date, radius=30, evl=False):
    #Example format place labels and coordinates. Not used by default script.
    labels = ["Leeds", "Sheffield", "York"]
    latlabels = [53.8008, 53.3811, 53.958]
    lonlabels = [-1.5491, -1.4701, -1.032]
    ##########
    plot_domain = Plot_Domain((-5.8, 1.9, 49.9, 55.9), labels=[], latlabels=[], lonlabels=[])
    zoom_domain = Plot_Domain((-1.9, -0.8, 53.4, 54.3), labels=labels,latlabels=latlabels, lonlabels=lonlabels)
    tnorm = mpl.colors.Normalize(vmin=0.0, vmax=24.0)

    def proc_rain(ffile):
        ccube = iris.load(ffile)[0]
        np.ma.masked_less(ccube.data, 0.01)
        return ccube

    def accums_mask(ffile):
        ccube = iris.load(ffile)[0]
        marray = np.ma.masked_less(ccube.data, 2.5)
        return marray.mask

    datadir = gws_root+"processed_forecasts/"+fcst_str[0:6]+"/"+fcst_str+"/"
    fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")

    date_str = "%04d%02d%02d" % (date.year, date.month, date.day)
    if not evl:
        figdir = user_root+"output_plots/forecasts/"+fcst_str[0:6]
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
        figdir = figdir + "/" + fcst_str
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)
    if evl:
        figdir = user_root+"output_plots/evaluation/"+date_str
        radardir = gws_root+"radar_obs/processed_radar/"+date_str+"_00/"
        if os.path.isdir(figdir) == False:
            os.system("mkdir " + figdir)

    try:
        ffile=datadir+date_str+'_mem_pp_r30_min_60_tot.nc'
        members=iris.load_cube(ffile).shape[0]
    except:
        if fcst_date.hour == 15:
            members=12
        else:
            members=18

    def plot_rwcrs(perc,evl=False):        
        mya=alphater()
        if not evl:
            f=MyPlot(2,2,3,plot_domain,date_str,fcst_str,figdir + "/" + date_str + "_{}_rwcrs".format(perc),figdir,dpi=500,dpi2=500,awidth=3,aheight=3,suptitle=True,lbotcol=False)
        if evl:
            f=MyPlot(3,2,6,plot_domain,date_str,fcst_str,figdir + "/" + fcst_str + "_{}_rwcrs".format(perc),figdir,dpi=500,dpi2=500,awidth=3,aheight=3,suptitle=True,lbotcol=True)
        f.start_bg()
        f.finish_bg()
        f.setup_decorations()
        f.append_subtitle(0,mya.next()+r') T60 ensemble')#, %02d:00, %02d/%02d/%04d'%(fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year))
        f.append_subtitle(1,mya.next()+r') T180 ensemble')
        f.append_subtitle(2,mya.next()+r') T360 ensemble')
        if not evl:
            f.add_legend("Rainfall (mm)\nCycle: %02dZ %02d/%02d/%04d\nValid: %02d/%02d/%04d"
                       %(fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year,date.day,date.month,date.year))
            f.add_suptitle('(r30,p%s) RWCRSs from MOGREPS-UK' %perc)
        if evl:
            f.append_subtitle(3,mya.next()+r') T60 radar')#, %02d:00, %02d/%02d/%04d'%(fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year))
            f.append_subtitle(4,mya.next()+r') T180 radar')
            f.append_subtitle(5,mya.next()+r') T360 radar')
            f.add_blegend("Rainfall (mm)")
            f.add_suptitle('%02d/%02d/%04d (r30,p%s) RWCRSs, fcst cycle %02dZ %02d/%02d/%04d'
                       %(date.day,date.month,date.year,perc,fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year) )
        f.process_decorations()
        f.setup_plots()
        ffile=datadir+date_str+'_ens_pp_r30_min_60_tot.nc'
        ccube=proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        f.append_image(0,ccube)
        ffile=datadir+date_str+'_ens_pp_r30_min_180_tot.nc'
        ccube=proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        f.append_image(1,ccube)
        ffile=datadir+date_str+'_ens_pp_r30_min_360_tot.nc'
        ccube=proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        f.append_image(2,ccube)
        if evl:
            ffile = radardir + date_str + "_00_rad_pp_r{}_min_60_tot.nc".format(radius)
            ccube = proc_rain(ffile)
            f.append_image(3, ccube.extract(iris.Constraint(percentile=perc)))
            ffile = radardir + date_str + "_00_rad_pp_r{}_min_180_tot.nc".format(radius)
            ccube = proc_rain(ffile)
            f.append_image(4, ccube.extract(iris.Constraint(percentile=perc)))
            ffile = radardir + date_str + "_00_rad_pp_r{}_min_360_tot.nc".format(radius)
            ccube = proc_rain(ffile)
            f.append_image(5, ccube.extract(iris.Constraint(percentile=perc)))
        f.process_figure()
        del f

    plot_rwcrs(98,evl=evl)

    def perc_max_accums(perc, period, radius=30, evl=False):
        if evl:
            num=members+2
            ffile=figdir + "/" + fcst_str + "_{}_max_T{}_accum".format(perc, period)
        else:
            num=members+1
            ffile=figdir + "/" + date_str + "_{}_max_T{}_accum".format(perc, period)
        mya = alphater()
        f = MyPlot(5,int((members+5)/5),num,plot_domain,date_str,fcst_str,ffile,figdir,suptitle=True,lbotcol=evl)
        # f.process_topo()
        f.start_bg()
        f.finish_bg()
        #os.system("cp output_plots/background_tiles/5_4_19_nat_ctopo.pdf "+ f.file_str+ "ctopo.pdf")
        #os.system("cp output_plots/background_tiles/5_4_19_nat_clines.pdf "+ f.file_str+ "clines.pdf")
        f.setup_decorations()
        if evl:
            f.append_subtitle(0, mya.next() + r") OBS")
            f.add_blegend("Rainfall (mm)")
        else:
            f.add_legend("Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"% (fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year,date.day,date.month,date.year) )
        for ii in range(1,members+1):
            f.append_subtitle(ii-(members+2-num), mya.next() + r") member " + str(ii))
        f.append_subtitle(num-1, mya.next() + r") ensemble")
        f.add_suptitle("(r%r,p%s,T%s) %02d/%02d/%04d %02dZ MOGREPS-UK cycle, valid %02d/%02d/%04d" % (radius,perc,period,fcst_date.day,fcst_date.month,fcst_date.year,fcst_date.hour,date.day,date.month,date.year))
        f.process_decorations()
        f.setup_plots()
        if evl:
            ffile = radardir + date_str + "_00_rad_pp_r{}_min_{}_tot.nc".format(radius,period)
            ccube = proc_rain(ffile)
            f.append_image(0, ccube.extract(iris.Constraint(percentile=perc)))
        ffile = datadir + date_str + "_mem_pp_r{}_min_{}_tot.nc".format(radius, period)
        ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        for ii in range(0,members):
            f.append_image(ii+(num-members-1), ccube[ii])
        ffile = datadir + date_str + "_ens_pp_r{}_min_{}_tot.nc".format(radius, period)
        ccube = proc_rain(ffile).extract(iris.Constraint(percentile=perc))
        f.append_image(num-1, ccube)
        f.process_figure()
        del f
    
    perc_max_accums(98,180,evl=evl)
    
    def exact_max_accums(period,evl=False):
        if evl:
            num=members+1
            ffile=figdir + "/" + fcst_str + "_exact_max_T{}_accum".format(period)
        else:
            num=members
            ffile=figdir + "/" + date_str + "_exact_max_T{}_accum".format(period)
        mya = alphater()
        f = MyPlot(5,int((members+4)/5),num,plot_domain,date_str,fcst_str,ffile,figdir,suptitle=True)
        # f.process_topo()
        f.start_bg()
        f.finish_bg()
        #os.system("cp output_plots/background_tiles/5_4_18_ctopo.pdf "+ f.file_str+ "ctopo.pdf")
        #os.system("cp output_plots/background_tiles/5_4_18_clines.pdf "+ f.file_str+ "clines.pdf")
        f.setup_decorations()
        f.add_legend("Rainfall (mm)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d" % (
            fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year,date.day,date.month,date.year))
        f.add_suptitle("%02d/%02d/%04d %02dZ MOGREPS-UK cycle, T%s max accums, valid %02d/%02d/%04d" % (fcst_date.day,fcst_date.month,fcst_date.year,fcst_date.hour,period,date.day,date.month,date.year))
        if evl:
            f.append_subtitle(0, mya.next() + r") OBS")
        for ii in range(1,members+1):
            f.append_subtitle(ii-(1+members-num), mya.next() + r") member " + str(ii))
        f.process_decorations()
        f.setup_plots()
        if evl:
            ffile = radardir + date_str + "_00_rad_exact_min_{}.nc".format(period)
            ccube = proc_rain(ffile)
            f.append_image(0, ccube[0])
        ffile = datadir + date_str + "_exact_min_{}.nc".format(period)
        ccube = proc_rain(ffile)
        for ii in range(0,members):
            f.append_image(ii+(num-members), ccube[ii])
        f.process_figure()
        del f

    exact_max_accums(180,evl=evl)

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
        f = MyPlot(5,4,18,plot_domain,date_str,fcst_str,figdir + "/" + date_str + "_exact_T{}_time".format(period),figdir,norm=tnorm,cmap=listed_ccmap,levels=[],suptitle=True)
        # f.process_topo()
        f.start_bg()
        f.finish_bg()
        f.setup_decorations()
        for ii in range(0, 18):
            f.append_subtitle(ii, mya.next() + r") member " + str(ii + 1))
        f.add_legend("Time (hr)\n\nForecast:\n%02dZ %02d/%02d/%04d\nValid:\n%02d/%02d/%04d"
            % (fcst_date.hour,fcst_date.day,fcst_date.month,fcst_date.year,date.day,date.month,date.year))
        f.add_suptitle("%02d/%02d/%04d exact forecast time of T%s max accum periods > 2.5mm" % (date.day, date.month, date.year, period))
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

    #exact_accum_time(180)

"""
def upload_plots(fcst_str):
    netdir = ("/gws/nopw/j04/icasp_swf/public/forecast_plots/" + fcst_str[0:6] + "/" + fcst_str)
    if os.path.isdir(netdir) == False:
        os.system("mkdir " + netdir)
    os.system("cp  /home/users/bmaybee/output_plots/forecasts/" + fcst_str[0:6] + "/" + fcst_str + "/* " + netdir)
"""

fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
days_ahead = []
if args.date is None:
    if fcst_date.year < 2019:
        out = 2
    else:
        out = 5
    for i in range(1, out):
        days_ahead.append(fcst_date + datetime.timedelta(days=i))
    
    nprocs = 16
    p = Pool(min(len(days_ahead), nprocs))
    p.map(plot_forecasts, days_ahead)
else:
    date=datetime.datetime.strptime(args.date, "%Y%m%d")
    plot_forecasts(date,evl=args.eval)

# upload_plots(fcst_str)
