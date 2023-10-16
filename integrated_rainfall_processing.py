# -*- coding: utf-8 -*-
from numba import njit, jit, prange, set_num_threads
from numba.types import bool_
import numpy as np
from glob import glob
import argparse
import iris
from iris.coords import DimCoord
from iris.time import PartialDateTime
import datetime
import time
import os
from utils import get_t_max, revamped_percentile_processing, make_and_save_cube


##################################################################
# Core script to generate RWCRS from calendar day rainfall fields, following method of Boeing et al (2020).
#
# Argument parser options:
# -f : %Y%m%d string of forecast initialisation date, or in case of radar field, date of observations. Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK)
# -d : If only need to process forecast for one calendar day, use this option. Takes %Y%m%d string of forecast validity date.
# -r : Default setting is to process ensemble forecasts. Provide non-empty argument to process radar field instead, in which case takes single calendar day's data (-f).
# -nc : Default setting is to act on rainfall netcdf files. Provide non-empty argument to use raw mogreps .pp files instead - highly recommended for national-scale domains.
# -swfhim : Needs -nc option. Default setting is to process 18 member MOGREPS ensembles. Provide non-empty argument to process 12 member ensembles, for comparison against SWFHIM legacy system.
# -w : Sets number of data timesteps included in processing window - default 5 * 12 = 60mins. Commented out for default FOREWARNS setting of [60, 180, 360] minute windows.
# -loc : Overarching data directory (see documentation of requisite folder structure).
#
# PREREQ's: mogreps_retrieval.sh OR radar_retrieval.sh
##################################################################

parser = argparse.ArgumentParser(
    prog="Percentile processor",
    description="A numba implementation of fast neighbourhood percentile processing",
)
parser.add_argument("-f", "--fcst_date", required=True, type=str)
parser.add_argument("-i", "--fcst_init", required=False, default=0, type=int)
parser.add_argument("-d", "--date", required=False, type=str)
parser.add_argument("-r", "--radar", required=False)
parser.add_argument("-nc", "--netcdf", required=False)
parser.add_argument("-swfhim", "--swfhim", required=False)
#parser.add_argument("-w", "--window_length", default=12, required=False, type=np.int32)
parser.add_argument("-loc", "--location", default='/gws/nopw/j04/icasp_swf/bmaybee/', required=False, type=str)
args = parser.parse_args()

fcst_str = args.fcst_date + "_%02d" % args.fcst_init
gws_root = args.location
if gws_root[-1] != "/":
    gws_root=gws_root+"/"
    
cubes = []
if args.radar is not None:
    glob_root = (gws_root + "radar_obs/" + fcst_str[:4] + "/" + fcst_str[:8])
    radar_file = glob_root + "_nimrod_ng_radar_rainrate_composite_1km_merged.nc"
    cubes.append(iris.load(radar_file)[0])
    # Nimrod radar resolution
    dgrid_km = 1
else:
    glob_root = (gws_root + "mogreps_ensembles/" + fcst_str[:6] + "/prods_op_mogreps-uk_" + fcst_str[:8])
    # MOGREPS-UK central domain resolution
    dgrid_km = 2.2

    # Processing netcdf vs pp filelists
    if args.netcdf is None:
        mogreps_files = glob_root + fcst_str[-3:] + "_??_merged.nc"
        mogreps_files = glob(mogreps_files)
        for member in range(len(mogreps_files)):
            cube = iris.load(mogreps_files[member])
            cubes.append(cube[0])
            del cube
    elif args.netcdf is not None:
        mems = np.arange(args.fcst_init-5, args.fcst_init+1)
        if args.swfhim is not None:
            mems = mems[-4:]
        zero_hr_list=[glob(glob_root + "_%02d_??_003.pp" % mem) for mem in mems]
        zero_hr_list=sum(zero_hr_list,[])
        example_files = [file[:-6]+"*.pp" for file in zero_hr_list]

        ##################################################################
        # MOGREPS-UK is a lagged ensemble forecast system. This code block forces all members to common time range
        # Time range is ( start time of latest initialised forecast - end time of earliest initialised foreacst )
        
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
        for file in glob(earliest_member_st+"*.pp"):
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
        timerange = iris.Constraint(time = lambda cell:
                                    PartialDateTime(year = en_st.year, month = en_st.month, day = en_st.day, hour = en_st.hour, minute = en_st.minute) 
                                    <= cell <=
                                    PartialDateTime(year = en_end.year, month = en_end.month, day = en_end.day, hour = en_end.hour, minute = en_end.minute))
    
        for member in range(len(example_files)):
            cube = iris.load(glob(example_files[member]))
            # Reduces domain from full MOGREPS -> England + Wales. EDIT IF DIFFERENT DOMAIN DESIRED.
            cube = cube[0][:,50:370,120:400]
            ##############
            cube = cube.extract(timerange)
            cubes.append(cube)
            del cube
            
        ##################################################################
        
print(fcst_str)

# Other settings
# Variables below are optimum parameters identified during verification over Northern England - see NHESS paper. Standard testbed settings, can be varied if desired.
percentiles = np.array([98])  # Percentiles to return
radii = np.array([30])

#percentiles = np.array([90, 95, 98, 99])
#radii = np.array([20,30,50])
minutes_per_timestep = 5
#minutes_in_window = minutes_per_timestep * args.window_length
# time_index_start = (12 * 16) - 1
# time_index_end = (12 * 40) - 2


def process_for_radius(
    radius,
    e_prec_t_max,
    member_dim,
    latitude_dim,
    longitude_dim,
    minutes_in_window,
):
    """
    Wrapper routine which sets up and saves the data cubes
    """
    # Search radius in grid points along longitude, does not need to be integer
    rad_i = radius / dgrid_km
    # Search radius in grid points along latitudes, does not need to be integer
    rad_j = radius / dgrid_km
    # Calculate ensemble post-processed rainfall, obtain indices of cropped domain
    ensemble_processed_data = revamped_percentile_processing(
        e_prec_t_max, rad_i, rad_j, percentiles
    )
    percentile_dim = DimCoord(np.double(percentiles), long_name="percentile", units="1")
    if args.radar is not None:
        make_and_save_cube(
            ensemble_processed_data,
            [percentile_dim, latitude_dim, longitude_dim],
            out_root
            + "_pp_r"
            + str(radius)
            + "_min_"
            + str(minutes_in_window)
            + "_tot.nc",
        )
        return

    else:
        make_and_save_cube(
            ensemble_processed_data,
            [percentile_dim, latitude_dim, longitude_dim],
            out_root
            + "_ens_pp_r"
            + str(radius)
            + "_min_"
            + str(minutes_in_window)
            + "_tot.nc",
        )
    del ensemble_processed_data
    # Process and save individual members
    num_members = np.shape(e_prec_t_max)[0]
    int_rad_i = int(np.ceil(rad_i))
    int_rad_j = int(np.ceil(rad_j))
    len_e = e_prec_t_max.shape[0]  # Number of ensemble members
    len_i = e_prec_t_max.shape[1]  # Number of lat indices
    len_j = e_prec_t_max.shape[2]  # Number of lon indices
    len_p = len(percentiles)
    # Store indices where to evaluate neighbourhood postprocessing
    # Ensure the same indices always selected for evaluation
    # independent of circle radius
    member_processed_data = np.zeros(
        (
            num_members,
            len_p,
            len_i,
            len_j,
        ),
        dtype=np.float32,
    )
    for member in range(num_members):
        member_processed_data[member] = revamped_percentile_processing(
            e_prec_t_max[member, :, :][None, :, :], rad_i, rad_j, percentiles
        )
    make_and_save_cube(
        member_processed_data,
        [member_dim, percentile_dim, latitude_dim, longitude_dim],
        out_root
        + "_mem_pp_r"
        + str(radius)
        + "_min_"
        + str(minutes_in_window)
        + "_tot.nc",
    )
    del member_processed_data


def process_files(day, minutes_in_window, cubes):
    """
    The main routine of the standalone script
    Sets up the data cubes and calls the optimal time and radius-dependent processing
    """
    date_str = "%04d%02d%02d" % (day.year, day.month, day.day)
    # Use first file to get latitudes and longitudes
    test_cube=cubes[0].copy()
    
    global out_root
    if args.radar is not None:
        out_root=gws_root+"radar_obs/processed_radar/"+fcst_str
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        out_root = out_root + "/" + fcst_str + "_rad"

        time_index_start, time_index_end = 0, -1
        latitude_dim = test_cube.coord("projection_y_coordinate")
        longitude_dim = test_cube.coord("projection_x_coordinate")
        len_lat = test_cube.shape[1]
        len_lon = test_cube.shape[2]
        # Nimrod radar units mm/hr - therefore apply standard conversion.
        seconds_per_timestep = minutes_per_timestep / 60

    else:
        out_root=gws_root+"processed_forecasts/"+fcst_str[:6]+"/"+fcst_str
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        out_root=out_root+"/"+date_str

        #Following lines restrict to calendar day. Need to modify if want to process different forecast period.
        hrs_ahead = int((day - fcst_stamp).total_seconds() / 3600)
        time_index_start = (12 * hrs_ahead) - 1
        time_index_end = time_index_start + 12 * 24 - 1
        latitude_dim = test_cube.coord("grid_latitude")
        longitude_dim = test_cube.coord("grid_longitude")
        len_lat = test_cube.shape[1]
        len_lon = test_cube.shape[2]
        # MOGREPS units kg m^3 / s - extra factor of 3600 to convert to mm/hr is wrapped into variable here (i.e. 1/60 * 3600) !
        seconds_per_timestep = minutes_per_timestep * 60
        
    del test_cube

    # Calculate and store optimal-time rainfall
    num_members = len(cubes)
    e_prec_t_max = np.zeros((num_members, len_lat, len_lon), dtype=np.float32)
    e_t_max_index = np.zeros((num_members, len_lat, len_lon), dtype=np.uint16)
    for member in range(num_members):
        member_cube = cubes[member]
        #member_cube = iris.load(example_files[member])[0]
        e_prec_t_max[member, :, :], e_t_max_index[member, :, :] = get_t_max(
            member_cube.data[time_index_start:time_index_end, :, :],
            # args.window_length,
            int(minutes_in_window / minutes_per_timestep),
            seconds_per_timestep,
        )
        del member_cube
    member_dim = DimCoord(
        np.arange(num_members, dtype=np.int32), long_name="realization", units="1"
    )

    # Add back start index when saving index cube
    make_and_save_cube(
        e_t_max_index + time_index_start,
        [member_dim, latitude_dim, longitude_dim],
        out_root + "_exact_min_" + str(minutes_in_window) + "_ind.nc",
        cube_type="index",
    )
    del e_t_max_index
    make_and_save_cube(
        e_prec_t_max,
        [member_dim, latitude_dim, longitude_dim],
        out_root + "_exact_min_" + str(minutes_in_window) + ".nc",
    )
    # Post-process for each radius
    for radius in radii:
        process_for_radius(
            radius,
            e_prec_t_max,
            member_dim,
            latitude_dim,
            longitude_dim,
            minutes_in_window,
        )


fcst_day = datetime.datetime.strptime(fcst_str[:8], "%Y%m%d")
fcst_stamp = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")

# Build range of dates for which to do processing.
if args.radar is not None:
    days_ahead = [fcst_day]
elif args.date is None:
    days_ahead = []
    #Prior to 2019 upgrade, MOGREPS max lead time = 52 hours, i.e. 2 complete calendar days
    if fcst_day.year < 2019:
        out = 2
    # Post 2019 upgrade, MOGREPS max lead time = 120 hours, i.e. 4 complete calendar days
    else:
        out = 5
    
    for i in range(1,out):
        days_ahead.append(fcst_day + datetime.timedelta(days=i)) 
else:
    days_ahead=[datetime.datetime.strptime(args.date, "%Y%m%d")]

t = time.time()
#Standard FOREWARNS accumulation periods. Need all three to generate flood forecast!
for period in [60, 180, 360]:
    for day in days_ahead:
        process_files(day, period, cubes)
print(time.time() - t)

# if __name__ == "__main__":
#    process_files()
