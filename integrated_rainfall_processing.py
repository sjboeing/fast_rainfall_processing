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

# Argument parser
parser = argparse.ArgumentParser(
    prog="Percentile processor",
    description="A numba implementation of fast neighbourhood percentile processing",
)
parser.add_argument("-f", "--fcst_date", required=True, type=str)
parser.add_argument("-i", "--fcst_init", required=False, default=0, type=int)
parser.add_argument("-d", "--date", required=False, type=str)
parser.add_argument("-r", "--radar", required=False)
parser.add_argument("-w", "--window_length", default=12, required=False, type=np.int32)
args = parser.parse_args()

fcst_str = args.fcst_date + "_%02d" % args.fcst_init
if args.radar is not None:
    glob_root = (
        "/gws/nopw/j04/icasp_swf/bmaybee/radar_obs/" + fcst_str[:4] + "/" + fcst_str[:8]
    )
    example_glob = glob_root + "_nimrod_ng_radar_rainrate_composite_1km_merged.nc"
    dgrid_km = 1

else:
    glob_root = (
        "/gws/nopw/j04/icasp_swf/bmaybee/mogreps_ensembles/"
        + fcst_str[:6]
        + "/prods_op_mogreps-uk_"
        + fcst_str
    )
    example_glob = glob_root + "_??_merged.nc"
    dgrid_km = 2.2
print(fcst_str)

# Input files
# glob_root = "/gws/nopw/j04/icasp_swf/bmaybee/mogreps_ensembles/202208/prods_op_mogreps-uk_20220815_08_"
# example_glob = glob_root + "??_merged.nc"
# example_files = glob(example_glob)

example_files = glob(example_glob)
##print(example_glob, example_files)

# Other settings
# percentiles = np.array([50, 90, 95, 98, 99, 99.5, 100])  # Percentiles to return
# radii = np.array([30, 40, 50])
percentiles = np.array([90, 95, 98, 99])
radii = np.array([30])
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


def process_files(day, minutes_in_window):
    """
    The main routine of the standalone script
    Sets up the data cubes and calls the optimal time and radius-dependent processing
    """
    date_str = "%04d%02d%02d" % (day.year, day.month, day.day)
    # Use first file to get latitudes and longitudes
    test_cube = iris.load(example_files[0])
    #print(test_cube[0])
    global out_root
    if args.radar is not None:
        """
        out_root = (
            "/home/users/bmaybee/manual_forecast_scripts/fast_rainfall_processing_files/"
            + fcst_str
        )
        """
        out_root="/gws/nopw/j04/icasp_swf/bmaybee/radar_obs/processed_radar/"+fcst_str
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        out_root = out_root + "/" + fcst_str + "_rad"

        time_index_start, time_index_end = 0, -1
        latitude_dim = test_cube[0].coord("projection_y_coordinate")
        longitude_dim = test_cube[0].coord("projection_x_coordinate")
        len_lat = test_cube[0].shape[1]
        len_lon = test_cube[0].shape[2]
        seconds_per_timestep = minutes_per_timestep / 60

    else:
        """
        out_root = (
            "/home/users/bmaybee/manual_forecast_scripts/fast_rainfall_processing_files/"
            + date_str
        )
        """
        out_root="/gws/nopw/j04/icasp_swf/bmaybee/processed_forecasts/"+fcst_str[:6]+"/"+fcst_str
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        out_root=out_root+"/"+date_str

        hrs_ahead = int((day - fcst_stamp).total_seconds() / 3600)
        time_index_start = (12 * hrs_ahead) - 1
        time_index_end = time_index_start + 12 * 24 - 1
        latitude_dim = test_cube[0].coord("grid_latitude")
        longitude_dim = test_cube[0].coord("grid_longitude")
        len_lat = test_cube[0].shape[1]
        len_lon = test_cube[0].shape[2]
        seconds_per_timestep = minutes_per_timestep * 60
    del test_cube

    # Calculate and store optimal-time rainfall
    num_members = len(example_files)
    e_prec_t_max = np.zeros((num_members, len_lat, len_lon), dtype=np.float32)
    e_t_max_index = np.zeros((num_members, len_lat, len_lon), dtype=np.uint16)
    for member in range(num_members):
        member_cube = iris.load(example_files[member])
        e_prec_t_max[member, :, :], e_t_max_index[member, :, :] = get_t_max(
            member_cube[0].data[time_index_start:time_index_end, :, :],
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

if args.radar is not None:
    days_ahead = [fcst_day]
elif args.date is None:
    days_ahead = []
    if fcst_day.year < 2019:
        out = 2
    else:
        out = 5
    
    for i in range(1,out):
        days_ahead.append(fcst_day + datetime.timedelta(days=i)) 
else:
    days_ahead=[datetime.datetime.strptime(args.date, "%Y%m%d")]

t = time.time()
for period in [60, 180, 360]:
    for day in days_ahead:
        process_files(day, period)
print(time.time() - t)

# if __name__ == "__main__":
#    process_files()
