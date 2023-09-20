# -*- coding: utf-8 -*-
#import gdal
# Note: on JASMIN use of ogr requires previous jaspy/3.7 installation. Install and use contextily to add basemaps easily.
import ogr
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shutil
#import contextily as cx
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import warnings
import datetime
warnings.filterwarnings("ignore")
import time

#standard options for libraries 
plt.ioff()
driver = ogr.GetDriverByName('ESRI Shapefile')
spatial_ref = 27700

##################################################################
# Core script to produce FOREWARNS flood forecast: conducts treshold look-up and produces basic map.
#
# Argument parser options:
# -f : %Y%m%d string of forecast initialisation date, or in case of radar the observations' date. Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK)
# -r : Default setting is to process ensemble forecasts. Provide non-empty argument to use radar field instead, in which case takes single calendar day's data (-f).
# -p : Percentile used to obtain underlying rainfall RWCRS. Default 98 identified through verification studies.
# -d : If only want plots for single calendar day, use this option. Takes %Y%m%d string of forecast validity date.
# -reg : Flood forecast region; corresponds to string in name of csv tables of catchment level flood threshold values. Current options are NEng or EngWls
# -loc : Overarching data directory (see documentation of requisite folder structure).
# -u : User directory (see documentation of requisite folder structure - where plots are outputed to).
#
# PREREQ's: extract_catchment_fcsts.py
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--forecastrun","-f",type=str,required=True)
parser.add_argument("--init","-i",type=str,default="08_",required=False)
parser.add_argument("--radar","-r",type=str,required=False)
parser.add_argument("--perc","-p",type=str,default="98",required=False)
parser.add_argument("--day","-d",type=int,required=False)
parser.add_argument("--reg","-reg",type=str,default="NEng",required=False)
parser.add_argument("--location","-loc",type=str,required=False,default="/gws/nopw/j04/icasp_swf/bmaybee/")
parser.add_argument("--user","-u",type=str,required=False,default="/home/users/bmaybee/")
args = parser.parse_args()
gws_root=args.location
user_root=args.user

if gws_root[-1] != "/":
    gws_root=gws_root+"/"
if user_root[-1] != "/":
    user_root=user_root+"/"

##universal params, for use in naming functions
years = [5,10,30,100,1000]
thresholdnames = ["5year","10year","30year","100year","1000year"]
durations = ['T60','T180','T360']
geoplots = gws_root+"flood_forecasts/Shapefile/GeoPlots_v1-00.shp"

###################
#Function to get all the forecast dates from the predicted files
def forecast_name_grab(forecasts):
    forecast_df = pd.read_csv(forecasts[0])
    no_cols = len(forecast_df)
    forecast_names = list(forecast_df.columns.values[3:no_cols])
    print(forecast_names)
    return(forecast_names)

###################
#Threshold look up - identifies return period associated with RWCRS rainfall field
def duration_threshold_test(forecasts,rainrefs,i):
    forecast_t60_df = pd.read_csv(forecasts[0])
    forecast_t180_df = pd.read_csv(forecasts[1])
    forecast_t360_df = pd.read_csv(forecasts[2])

    rainfall_t60_df = pd.read_csv(rainrefs[0])
    rainfall_t180_df = pd.read_csv(rainrefs[1])
    rainfall_t360_df = pd.read_csv(rainrefs[2])
    
    forecast_t60_col = forecast_t60_df.iloc[:,[0,3+i]]
    forecast_t180_col = forecast_t180_df.iloc[:,[0,3+i]]
    forecast_t360_col = forecast_t360_df.iloc[:,[0,3+i]]
    
    s_t60_6,s_t60_10,s_t60_30,s_t60_100,s_t60_1000 = fast_all_threshold_test(forecast_t60_col,rainfall_t60_df)
    s_t180_6,s_t180_10,s_t180_30,s_t180_100,s_t180_1000 = fast_all_threshold_test(forecast_t180_col,rainfall_t180_df)
    s_t360_6,s_t360_10,s_t360_30,s_t360_100,s_t360_1000 = fast_all_threshold_test(forecast_t360_col,rainfall_t360_df)
    
    s_6_res = [s_t60_6,s_t180_6,s_t360_6]
    s_10_res = [s_t60_10,s_t180_10,s_t360_10]
    s_30_res = [s_t60_30,s_t180_30,s_t360_30]
    s_100_res = [s_t60_100,s_t180_100,s_t360_100]
    s_1000_res = [s_t60_1000,s_t180_1000,s_t360_1000]
    
    s6 = list(set().union(*s_6_res))
    s10 = list(set().union(*s_10_res))
    s30 = list(set().union(*s_30_res))
    s100 = list(set().union(*s_100_res))
    s1000 = list(set().union(*s_1000_res))
    
    #print(len(s6),len(s10),len(s30),len(s100),len(s1000))
    s6,s10,s30,s100,s1000 = sort_all_catchments(s6,s10,s30,s100,s1000)
    #if len(s6) == 0 and len(s1000) > 0:
    #    s6.append(s10[0])
    #print(s6,s10,s30,s100,s100)
    #print(len(s6),len(s10),len(s30),len(s100),len(s1000))
    #print(len(s6)+len(s10)+len(s30)+len(s100)+len(s1000),len(forecast_t60_df))
    return(s6,s10,s30,s100,s1000)

# Look-up comparison for 30 year and above thresholds
def fast_high_threshold_test(forecast_col,rainfall_df):
    forecast_col=forecast_col.rename(columns = {list(forecast_col)[1]:'rain'})
    s30 = forecast_col["fid"][(forecast_col["rain"].ge(rainfall_df["30year"])) & (forecast_col["rain"].le(rainfall_df["100year"]))]
    s100 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["100year"]) & (forecast_col["rain"] < rainfall_df["1000year"])]
    s1000 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["1000year"])]
    return(s30.values,s100.values,s1000.values)

# Look-up comparison for 6 (5) year and above thresholds
def fast_all_threshold_test(forecast_col,rainfall_df):
    forecast_col=forecast_col.rename(columns = {list(forecast_col)[1]:'rain'})
    s6 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["6year"]) & (forecast_col["rain"] < rainfall_df["10year"])]
    s10 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["10year"]) & (forecast_col["rain"] < rainfall_df["30year"])]
    s30 = forecast_col["fid"][(forecast_col["rain"].ge(rainfall_df["30year"])) & (forecast_col["rain"].le(rainfall_df["100year"]))]
    s100 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["100year"]) & (forecast_col["rain"] < rainfall_df["1000year"])]
    s1000 = forecast_col["fid"][(forecast_col["rain"] > rainfall_df["1000year"])]
    #if len(s1000) > 0:
    #    print(forecast_col[forecast_col["fid"].isin(s100.values)],rainfall_df[["fid","100year"]][rainfall_df["fid"].isin(s100.values)])
    #print(s30.values,s100.values,s1000.values)
    return(s6.values,s10.values,s30.values,s100.values,s1000.values)

#Below functions identify unique (maximum) return period associated with each catchment location
def sort_high_catchments(s30,s100,s1000):
    thresholdbins = [s30,s100,s1000]
    catchs = []
    #first unique catchments only
    for i,r in enumerate(thresholdbins):
        r = list(set(r))
        catchs.append(r)
    ##sort s30 first - remove any instances of s30 being in s100 and s1000
    s30 = [x for x in catchs[0] if x not in catchs[1]]
    s30 = [x for x in s30 if x not in catchs[2]]
    ##sort s100 - remove any instances of s100 being in s1000
    s100 = [x for x in catchs[1] if x not in catchs[2]]
    s1000 = catchs[2]
    return (s30,s100,s1000)

def sort_all_catchments(s6,s10,s30,s100,s1000):
    thresholdbins = [s6,s10,s30,s100,s1000]
    catchs = []
    #first unique catchments only
    for i,r in enumerate(thresholdbins):
        r = list(set(r))
        catchs.append(r)
    s6 = [x for x in catchs[0] if x not in catchs[1]]
    s6 = [x for x in s6 if x not in catchs[2]]
    s6 = [x for x in s6 if x not in catchs[3]]
    s6 = [x for x in s6 if x not in catchs[4]]
    s10 = [x for x in catchs[1] if x not in catchs[2]]
    s10 = [x for x in s10 if x not in catchs[3]]
    s10 = [x for x in s10 if x not in catchs[4]]
    s30 = [x for x in catchs[2] if x not in catchs[3]]
    s30 = [x for x in s30 if x not in catchs[4]]
    s100 = [x for x in catchs[3] if x not in catchs[4]]
    s1000 = catchs[4]
    return (s6,s10,s30,s100,s1000)

#################################
# Translate results of threshold look ups into geospatial data - towards building foreacst plot
def select_catchs(catchmentfile,folder,select6,select10,select30,select100,select1000,reg):
    catchmentshp = ogr.Open(catchmentfile)
    catchmentlyr = catchmentshp.GetLayer()
    layer=catchmentlyr
    #for i in range(len(layer.schema)):
    #    print(layer.schema[i].name)
    mylist=[]
    for feature in layer:
        mylist.append(feature.GetField('fid'))
    thresholdbins = [select6,select10,select30,select100,select1000]
    for i,rp in enumerate(thresholdbins):
        thresholdyear = thresholdnames[i]
        outShapefile = os.path.join(folder,str(thresholdyear+"_{}_mask.shp".format(reg)))
        if os.path.exists(outShapefile):
            os.remove(outShapefile)
        outDataSource = driver.CreateDataSource(outShapefile)
        out_lyr_name = os.path.splitext(os.path.split(outShapefile )[1] )[0]
        outLayer = outDataSource.CreateLayer(out_lyr_name, geom_type=ogr.wkbMultiPolygon )
        outLayer.CreateField(ogr.FieldDefn("RP", ogr.OFTInteger))
        outLayer.CreateField(ogr.FieldDefn("CatchID", ogr.OFTInteger))
        # NOTE: this loop produces the warning "ERROR 1: Invalid index : -1". THIS IS NOT AN ERROR. Seeing this in terminal shows flood risk has been allocated. Haven't been able to turn off....
        for fid_value in rp:
            feat = catchmentlyr.GetFeature(mylist.index(fid_value))
            out_feat = ogr.Feature(catchmentlyr.GetLayerDefn())
            out_feat.SetGeometry(feat.GetGeometryRef().Clone())
            out_feat.SetField("RP",int(fid_value))
            out_feat.SetField("CatchID",int(fid_value))
            outLayer.CreateFeature(out_feat)
            outLayer.SyncToDisk()
            feat.Destroy()
            out_feat = None
    outDataSource.Destroy()
    outDataSource = None
    return()
    
###########################
##merge shapefiles - make basic file underpinning flood plots (outputs stored in Shapefile; can be extracted and used in eg QGIS to make nicer output maps easily).
def update_shp(thersholdnames,years,colname,s6,s10,s30,s100,s1000,reg,fcst_str=None):
    thresholdbins = [s6,s10,s30,s100,s1000]
    for i,thresholdyear in enumerate(thresholdnames):
        thresholdval = years[i]
        threshbin = thresholdbins[i]
        threshshpfile = os.path.join(processfolder,str(thresholdyear+"_{}_mask.shp".format(reg)))
        tshp = gpd.read_file(threshshpfile)
        if not tshp.empty:
            tshp.RP = thresholdval
            tshp.CatchID = threshbin
            tshp.to_file(threshshpfile)
    cat6 = gpd.read_file(os.path.join(processfolder,str("5year_{}_mask.shp".format(reg))))
    cat10 = gpd.read_file(os.path.join(processfolder,str("10year_{}_mask.shp".format(reg))))
    cat30 = gpd.read_file(os.path.join(processfolder,str("30year_{}_mask.shp".format(reg))))
    cat100 = gpd.read_file(os.path.join(processfolder,str("100year_{}_mask.shp".format(reg))))
    cat1000 = gpd.read_file(os.path.join(processfolder,str("1000year_{}_mask.shp".format(reg))))
    floodcats = gpd.GeoDataFrame(pd.concat([cat6,cat10,cat30,cat100,cat1000]))
    floodcats.to_file(os.path.join(processfolder,str(colname+"_FloodCatchments.shp")))
    #if fcst_str is not None:
    #    floodcats.to_file(processfolder+"/../workshop_event_data/"+colname[:8]+"/"+fcst_str+"_"+colname[-8:]+"_FloodCatchments.shp")
    return()
    
################
#plot basic regional floodmap. Domain set automatically by catchment shapefiles.
def plotflood(catchment_file,title_data,floodcatsfile,directory,perc,urban_csv=None):
    #urban_p = pd.read_csv(urban_csv)
    floodcats = gpd.read_file(floodcatsfile)
    all_cat_gdf = gpd.read_file(catchment_file)
    geop = gpd.read_file(geoplots)
    floodcats = geop.append(floodcats)
    floodcats.RP = floodcats.RP.astype(float)
    #urbanareas = gpd.read_file(urban_shp)
    minx, miny, maxx, maxy = all_cat_gdf.geometry.total_bounds
    ax = all_cat_gdf.plot(figsize=(10, 10),color='none',edgecolor='black')
    #urbanareas.plot(ax=ax,marker = 'o',color='red', markersize=50)
    #ax = floodcats.plot(figsize=(10, 10), alpha=0.5, edgecolor='k',column="RP",legend="True",categorical="True")
    #ax.set_xlim(minx - .1, maxx + .1)
    #ax.set_ylim(miny - .1, maxy + .1)
    ax.axis('off')
    fcst_date=datetime.datetime.strptime(title_data[0],"%Y%m%d_%H")
    #fcst_date=fcst_date + datetime.timedelta(hours=4)
    date=datetime.datetime.strptime(title_data[1][:8],"%Y%m%d")
    if title_data[2]=="ens":
        title_str="(r30,p"+title_data[3]+") FOREWARNS forecast valid %02d/%02d/%04d"%(date.day,date.month,date.year)+"\nFrom %02d/%02d/%04d %02dUTC MOGREPS-UK cycle"%(fcst_date.day,fcst_date.month,fcst_date.year,fcst_date.hour) 
        file_str=os.path.join(directory,str(title_data[1][:8]+"_fcst_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    elif title_data[2]=="rad":
        title_str="(r30,p"+title_data[3]+") FOREWARNS radar benchmarking, %02d/%02d/%04d "%(fcst_date.day,fcst_date.month,fcst_date.year)
        file_str=os.path.join(directory,str(title_data[1][:8]+"_rad_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    else:
        title_str=""
        file_str=os.path.join(directory,str(title_data[0]+"_"+title_data[2]+"_fcst_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    ax.set_title(title_str)
    leg = floodcats.plot(ax = ax,alpha=0.5, edgecolor='k',column="RP",cmap="viridis",categorical="True", legend="True")
    legend_labels = leg.get_legend().get_texts()
    for bound, legend_label in zip(years, legend_labels):
        legend_label.set_text(bound)
    """
    for index,row in urban_p.iterrows():
        x = row.X
        y = row.Y
        city = row.City
        ax.annotate(city, xy=(x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize = 20,color="black",bbox=dict(facecolor='b', alpha=0.2))
    """
    #cx.add_basemap(ax, crs=all_cat_gdf.crs)
    ax.figure.savefig(file_str,bbox_inches='tight')
    
#######################
def plotnoflood(catchment_file,title_data,directory,perc,urban_csv=None):
    #urban_p = pd.read_csv(urban_csv)
    all_cat_gdf = gpd.read_file(catchment_file)
    #urbanareas = gpd.read_file(urban_shp)
    minx, miny, maxx, maxy = all_cat_gdf.geometry.total_bounds
    ax = all_cat_gdf.plot(figsize=(10, 10),color='none',edgecolor='black')
    #urbanareas.plot(ax=ax,marker = 'o',color='red', markersize=50)
    #ax = floodcats.plot(figsize=(10, 10), alpha=0.5, edgecolor='k',column="RP",legend="True",categorical="True")
    #ax.set_xlim(minx - .1, maxx + .1)
    #ax.set_ylim(miny - .1, maxy + .1)
    ax.axis('off')
    fcst_date=datetime.datetime.strptime(title_data[0],"%Y%m%d_%H")
    #fcst_date=fcst_date + datetime.timedelta(hours=4)
    date=datetime.datetime.strptime(title_data[1][:8],"%Y%m%d")
    if title_data[2]=="ens":
        title_str="(r30,p"+title_data[3]+") FOREWARNS forecast valid %02d/%02d/%04d"%(date.day,date.month,date.year)+"\nFrom %02d/%02d/%04d %02dUTC MOGREPS-UK cycle"%(fcst_date.day,fcst_date.month,fcst_date.year,fcst_date.hour) 
        file_str=os.path.join(directory,str(title_data[1][:8]+"_fcst_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    elif title_data[2]=="rad":
        title_str="(r30,p"+title_data[3]+") FOREWARNS radar benchmarking, %02d/%02d/%04d "%(fcst_date.day,fcst_date.month,fcst_date.year)
        file_str=os.path.join(directory,str(title_data[1][:8]+"_rad_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    else:
        title_str=""
        file_str=os.path.join(directory,str(title_data[0]+"_"+title_data[2]+"_fcst_"+perc+"_{}_Floodplots.jpg".format(title_data[-1])))
    ax.set_title(title_str)
    
    """
    for index,row in urban_p.iterrows():
        x = row.X
        y = row.Y
        city = row.City
        ax.annotate(city, xy=(x, y), xytext=(3, 3), textcoords="offset points",
                    fontsize = 20,color="black",bbox=dict(facecolor='b', alpha=0.2))
    """
    #cx.add_basemap(ax, crs=all_cat_gdf.crs)
    ax.figure.savefig(file_str,bbox_inches='tight')

"""
#These functions get RoSWF images for areas highlighted as having flood risk. Slow and produce a lot of outputs.
#######################
def combine_images(forecast_dir,forecastname,forecast_names,member):
##read the images in after creation, to create a single image for use in the flood forecast report
    forecast_images = [os.path.join(forecast_dir,str(x+"_Floodplots.jpg")) for x in forecast_names]
    images = [Image.open(x) for x in forecast_images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]   
    new_im.save(os.path.join(forecast_dir,str(forecastname+"_"+member+"_FloodForecast.jpg")))
    return()
    
################# 
##select flood tiles from the referece map, then create an event footprint from the catchments
def selecttiles(select30,select100,select1000,flood_dir,forecast_dir,thresholdnames):
    thresholdbins = [select30,select100,select1000]
    for i,rp in enumerate(thresholdbins):
        for fid_value in rp:
            rp_points = thresholdnames[i]
            image = os.path.join(flood_dir,str("Catchment"+str(fid_value)+"_"+rp_points+"_Floodplots.jpg"))
            shutil.copy2(image, forecast_dir)
    return()
"""

def main():
    forecastrun = args.forecastrun
    perc = args.perc
    init = args.init
    reg = args.reg
    init=init+"_"
    global processfolder, outputfolder
    processfolder = gws_root+"flood_forecasts/Shapefile"
    outputfolder = os.path.join(user_root,"output_plots")
    if args.radar is not None:
        init="00_"
        member="rad"
        outputfolder = outputfolder+'/radar'
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        forecast_dir = os.path.join(outputfolder,forecastrun[:6])
        if not os.path.exists(forecast_dir):
            os.makedirs(forecast_dir)
        data_root = gws_root+"flood_forecasts/rainfall_inputs/radar/"+forecastrun
    else:
        member="ens"
        outputfolder = outputfolder+"/forecasts/"+forecastrun[:6]
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        forecast_dir = os.path.join(outputfolder,forecastrun+"_"+init[:-1])
        if not os.path.exists(forecast_dir):
            os.makedirs(forecast_dir)
        data_root = gws_root+"flood_forecasts/rainfall_inputs/"+forecastrun
 
    if args.init is None and args.radar is None:
        init="15_"        
        
    catchment_file = gws_root+"flood_forecasts/Shapefile/{}_Catchments_v1-00.shp".format(reg)
    
    forecasts = [os.path.join(data_root,str(forecastrun+"_"+init+member+"_"+perc+"_fcst_{}_catchment_max_".format(reg)+x+"_accums_r30.csv")) for x in durations]
    #CHANGE TO _accurate WHEN HAVE TRUE DATA:
    refs="_dummy"
    rainrefs = [str(gws_root+"flood_forecasts/RainfallReferenceFiles/{}_HB_30km_CatchmentRainfall_".format(reg)+x+refs+".csv") for x in durations]
    forecast_names = forecast_name_grab(forecasts)
    forecast_names = [name+"_"+member for name in forecast_names]
    if args.day is not None:
        forecast_names = [forecast_names[args.day]]
        
    print(forecast_names)
    for index in range(0,len(forecast_names)):
        forecast_name = forecast_names[index]
        floodcatsfile = (os.path.join(processfolder,str(forecast_name+"_FloodCatchments.shp")))
        if args.day is not None:
            s6,s10,s30,s100,s1000 = duration_threshold_test(forecasts,rainrefs,args.day)
        else:
            s6,s10,s30,s100,s1000 = duration_threshold_test(forecasts,rainrefs,index)
        if len(s6) == 0 and len(s10) == 0 and len(s30) == 0 and len(s100) == 0 and len(s1000) == 0:
            print(str("No Surface Water Events: "+forecast_name))
            plotnoflood(catchment_file,[forecastrun+"_"+init[:-1],forecast_name,member,perc,reg],forecast_dir,perc)
        else:
            select_catchs(catchment_file,processfolder,s6,s10,s30,s100,s1000,reg)
            update_shp(thresholdnames,years,forecast_name,s6,s10,s30,s100,s1000,reg,forecastrun+"_"+init[:-1])
            plotflood(catchment_file,[forecastrun+"_"+init[:-1],forecast_name,member,perc,reg],floodcatsfile,forecast_dir,perc)
    print("Processing Complete")
    
if __name__== '__main__':
    main()