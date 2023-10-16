import iris
from iris.time import PartialDateTime
import datetime
import glob
from multiprocessing import Pool
import time
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fcst_date", required=True, type=str)
parser.add_argument("-i", "--fcst_init", required=True, type=int)
#Get names of each ensemble member from earliest lead-time forecasts

args = parser.parse_args()
datadir="/gws/nopw/j04/icasp_swf/bmaybee/mogreps_ensembles/"+args.fcst_date[:6]
print(datadir)
if args.fcst_init == 15:
    mems = [12,13,14,15]
else:
    mems = np.arange(args.fcst_init-5, args.fcst_init+1)
zero_hr_list=[glob.glob(datadir + "/*" + args.fcst_date +"_%02d_??_003.pp" % mem) for mem in mems]
zero_hr_list=sum(zero_hr_list,[])
#zero_hr_list=glob.glob(datadir+"/*"+args.fcst_date+"*003.pp")
nprocs=32
print(len(zero_hr_list))

#MOGREPS ensemble members have different initialisation times! Need to account for this.
#All members will need to be cropped to start from start time of latest-initialised member, and to end at end time of earliest-initialised member
#Get name of latest initialised member (1 of the 3, doesn't matter which)
max_hr = 0
min_hr = 10e7
for file in zero_hr_list:
    init = iris.load(file)[0].coord("forecast_reference_time").points
    if init < min_hr:
        min_hr = init
        earliest_member_st = file[:-7]
    if init > max_hr:
        max_hr = init
        latest_member_st = file

#Get name of latest lead-time forecast generated from earliest initialised member (available up to 5 days after initialisation)
m_idx = 0
for file in glob.glob(earliest_member_st+"*.pp"):
    m = int(file[-6:-3].lstrip("0"))
    if m > m_idx:
        earliest_member_end = file
        m_idx = m

#Access start and end times of latest and earliest initialised members respectively        
ensemble_start = iris.load(latest_member_st)[0].coord("time")[0]
ensemble_end = iris.load(earliest_member_end)[0].coord("time")[-1]
#Convert to datetime
en_st = ensemble_start.units.num2date(ensemble_start.points)[0]
en_end = ensemble_start.units.num2date(ensemble_end.points)[0]

#Merge all files belonging to given ensemble member
def process_forecast(zero_hr_file):
    basename=zero_hr_file[:-7]
    print(basename)
    cube = iris.load(basename+"*.pp")
    #Crop time-span to that identified above; this step requires jaspy/3.7!!!
    timerange = iris.Constraint(time = lambda cell:
                                PartialDateTime(year = en_st.year, month = en_st.month, day = en_st.day, hour = en_st.hour, minute = en_st.minute) 
                                <= cell <=
                                PartialDateTime(year = en_end.year, month = en_end.month, day = en_end.day, hour = en_end.hour, minute = en_end.minute))
    cube = cube.extract(timerange)
    #cube = cube[0][:,50:370,120:400]
    cube = cube[0][:,50:100,120:160]
    outcubename=latest_member_st[:-10] + basename[-3:] + "_merged.nc"
    iris.save(cube, 
              outcubename)#,netcdf_format='NETCDF4', zlib=True, complevel=4,least_significant_digit=6)
    
    return cube, outcubename


st=time.time()    
if __name__ == '__main__':
    p = Pool(min(nprocs,len(zero_hr_list)))
    outputs = p.map(process_forecast, zero_hr_list)
print(time.time()-st)
