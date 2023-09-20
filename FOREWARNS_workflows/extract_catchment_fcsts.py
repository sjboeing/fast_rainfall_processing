from plot_utils import *
import xarray as xr
import pandas as pd
import glob
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import argparse

##################################################################
# Core script to extract RWCRS rainfall accumulations at catchment centroids.
# IMPORTANT: for time efficiency, true values are only extracted if the maximum RWCRS field value is greater than the minimum catchment return period threshold. Otherwise filled with 1's
# This makes no difference to the final flood foreacst, but hugely reduces the time spent processing days with no identified flood risk. Step indiciated below if removal desired.
#
# Argument parser options:
# -f : %Y%m%d string of forecast initialisation date, or in case of radar the observations' date. Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK)
# -r : Default setting is to process ensemble forecasts. Provide non-empty argument to use radar field instead, in which case takes single calendar day's data (-f).
# -n : Neighbourhood radius. Default 30km identified through verification studies.
# -d : If only want plots for single calendar day, use this option. Takes %Y%m%d string of forecast validity date.
# -reg : Flood forecast region; corresponds to string in name of csv tables of catchment level flood threshold values. Current options are NEng or EngWls
# -loc : Overarching data directory (see documentation of requisite folder structure).
#
# PREREQ's: integrated_rainfall_processing.py
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fcst_str", required=True, type=str)
parser.add_argument("-i", "--fcst_init", required=False, default=0, type=str)
parser.add_argument("-r", "--radar", required=False)
parser.add_argument("-n", "--nhood", required=False, default=30, type=int)
parser.add_argument("-d", "--date", required=False, type=str)
parser.add_argument("-reg", "--region", default="EngWls", required=False)
parser.add_argument("-loc", "--location", default="/gws/nopw/j04/icasp_swf/bmaybee", required=False, type=str)
args = parser.parse_args()

gws_root=args.location
if gws_root[-1] != "/":
    gws_root=gws_root+"/"
fcst_str=args.fcst_str+"_"+(args.fcst_init).zfill(2)
fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
radius=args.nhood
dates = []
if args.radar is not None:
    dates.append(fcst_date)
    mem='rad'
elif args.date is not None:
    dates.append(datetime.datetime.strptime(args.date, "%Y%m%d"))
    mem='ens'
else:
    if fcst_date.year < 2019:
        end=2
    else:
        end=5
    for i in range(1,end):
        dates.append(fcst_date + datetime.timedelta(days=i))
    mem='ens'

print(dates,mem)

#Get catchment centroid locations from reference csv table. Coordinates are UK Ordnance Survey National Grid rather than lat-lon.
df = pd.read_csv(gws_root+'/flood_forecasts/RainfallReferenceFiles/{}_HB_30km_catchments.csv'.format(args.region))
df = df[['fid','X','Y']]
# CHANGE TO _accurate FOR TRUE LOOK-UP VALUES (only used here for the determining whether to pad-fill with 1's).
refs = "dummy"

def extract_OSGB_hyetograph(ffile,lon,lat,perc):
    rainfall=iris.load(ffile)[0]
    try:
        rainfall=rainfall.extract(iris.Constraint(percentile=perc))
    except:
        pass
    
    try: 
        rot_pole = rainfall.coord('grid_latitude').coord_system.as_cartopy_crs()  
        ll = ccrs.OSGB()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [('grid_latitude', target_xy[1]), ('grid_longitude', target_xy[0])]
    except:
        rot_pole = rainfall.coord('projection_y_coordinate').coord_system.as_cartopy_crs()        
        ll = ccrs.OSGB()
        target_xy = rot_pole.transform_point(lon, lat, ll)
        sample_points = [('projection_y_coordinate', target_xy[1]), ('projection_x_coordinate', target_xy[0])]
    rainfall_series=rainfall.interpolate(sample_points, iris.analysis.Nearest())
    return rainfall_series.data
    
def process_period_forecasts(period):
    fcst_str = "%04d%02d%02d_%02d"%(fcst_date.year,fcst_date.month,fcst_date.day,fcst_date.hour)
    date_str = "%04d%02d%02d"%(date.year,date.month,date.day)
    df_ids = df.copy()
    # Get minimum cathchment rainfall threshold -- NOTE USE OF DUMMY VALUES:
    rain_ref=pd.read_csv(gws_root+"/flood_forecasts/RainfallReferenceFiles/{}_HB_30km_CatchmentRainfall_T{}_{}.csv".format(args.region,period,refs))['6year'].min()
        
    if args.radar is not None:
        ffile=gws_root+"/radar_obs/processed_radar/"+fcst_str+"/"+fcst_str+"_rad_pp_r{}_min_{}_tot.nc".format(radius,period)
        outdir=gws_root+"/flood_forecasts/rainfall_inputs/radar/"+fcst_str[:-3]
    else:
        ffile=gws_root+"/processed_forecasts/"+fcst_str[:6]+"/"+fcst_str+"/"+date_str+"_ens_pp_r{}_min_{}_tot.nc".format(radius,period)
        outdir=gws_root+"/flood_forecasts/rainfall_inputs/"+fcst_str[:-3]
    fcst_ref=xr.open_dataset(ffile).rainfall_amount.sel(percentile=perc)
    # Get maximum RWCRS field value; filter applied in case of appearance of erroneous infinite values (seen in radar):
    fcst_ref=float(fcst_ref.where(fcst_ref<10000).max())

    ##################################################################
    # Part of code which fills data with 1 in case of no identified flood risk.
    if fcst_ref > rain_ref:
        #print(date_str,period,perc,": thresholds breached, extracting values")
        try:
            df_ids[date_str+"_fcst"] = df_ids.apply(lambda x: extract_OSGB_hyetograph(ffile,x["X"],x["Y"],perc), axis = 1)
            df_ids[date_str+"_fcst"] = df_ids[date_str+"_fcst"].where(df_ids[date_str+"_fcst"]<10000)
        except:
            print(date_str+" error; accurate flood forecast not possible")
            df_ids[date_str+"_fcst"] = 100
    else:
        #print(date_str,period,perc,": no thresholds breached")
        df_ids[date_str+"_fcst"] = 1
    ##################################################################
    
    df_ids.to_csv(outdir+"/"+fcst_str+"_"+mem+"_%02d_fcst_{}_catchment_max_T%s_accums_r%02d.csv".format(args.region) %(perc,period,radius),index=False)
                  
def process_date_forecasts(date):
    fcst_str = "%04d%02d%02d_%02d"%(fcst_date.year,fcst_date.month,fcst_date.day,fcst_date.hour)
    date_str = "%04d%02d%02d"%(date.year,date.month,date.day)
    df_ids = df.copy()
    rain_ref=pd.read_csv(gws_root+"/flood_forecasts/RainfallReferenceFiles/{}_HB_30km_CatchmentRainfall_T{}_{}.csv".format(args.region,period,refs))['6year'].min()
        
    if args.radar is not None:
        ffile=gws_root+"/radar_obs/processed_radar/"+fcst_str+"/"+fcst_str+"_rad_pp_r{}_min_{}_tot.nc".format(radius,period)
    else:
        ffile = gws_root+"/processed_forecasts/"+fcst_str[:6]+"/"+fcst_str+"/"+date_str+"_ens_pp_r{}_min_{}_tot.nc".format(radius,period)
    fcst_ref=xr.open_dataset(ffile).rainfall_amount.sel(percentile=perc)
    #filter applied in case of appearance of erroneous infinite values (seen in radar)
    fcst_ref=float(fcst_ref.where(fcst_ref<10000).max())
    if fcst_ref > rain_ref:
        #print(date_str,period,perc,": thresholds breached, extracting values")
        try:
            df_ids[date_str+"_fcst"] = df_ids.apply(lambda x: extract_OSGB_hyetograph(ffile,x["X"],x["Y"],perc), axis = 1)
        except:
            print(date_str+" error; accurate flood forecast not possible")
            # Value of 100 indicates an error has occured!
            df_ids[date_str+"_fcst"] = 100
    else:
        #print(date_str,period,perc,": no thresholds breached")
        # A value of 1 DOES NOT indicated an error has occured; rather, no thresholds were breached.
        df_ids[date_str+"_fcst"] = 1

    return df_ids[date_str+"_fcst"]

#FOREWARNS standard parameters
periods=[60,180,360]
percs=[98]

#In case of multiple dates, combine each day's column of look-up values into single csv file
if len(dates)>1:
    for perc in percs:
        for period in periods:
            p = Pool(len(dates))
            pop_columns = p.map(process_date_forecasts, dates)
            pop_columns.insert(0,df)
            precip_vals=pd.concat(pop_columns,axis=1)
            cols=precip_vals.columns.tolist()
            for i in range(5,len(cols)):
                if int(cols[i-2][:8]) > int(cols[i-1][:8]):
                    cols=cols[:i-2]+cols[i-1]+cols[i-2]+cols[i:]
                else:
                    continue
            precip_vals=precip_vals[cols]
    
            if args.radar is not None:
                outdir=gws_root+"/flood_forecasts/rainfall_inputs/radar/"+fcst_str[:-3]
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
            else:
                outdir=gws_root+"/flood_forecasts/rainfall_inputs/"+fcst_str[:-3]
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
            precip_vals.to_csv(outdir+"/"+fcst_str+"_"+mem+"_%02d_fcst_{}_catchment_max_T%s_accums_r%02d.csv".format(args.region) %(perc,period,radius),index=False)

else:
    date=dates[0]
    if args.radar is not None:
        outdir=gws_root+"/flood_forecasts/rainfall_inputs/radar/"+fcst_str[:-3]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir=gws_root+"/flood_forecasts/rainfall_inputs/"+fcst_str[:-3]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            
    for perc in percs:
        p = Pool(len(periods))
        pop_columns = p.map(process_period_forecasts, periods)