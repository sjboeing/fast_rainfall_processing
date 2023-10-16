"""This module contains a stand-alone version of the postprocessing algorithm."""
from glob import glob
import argparse
import numpy as np
import iris
from iris.coords import DimCoord
from utils import get_t_max, revamped_percentile_processing, make_and_save_cube

# Argument parser
parser = argparse.ArgumentParser(
    prog="Percentile processor",
    description="A numba implementation of fast neighbourhood percentile processing",
)

parser.add_argument("-w", "--window_length", default=12, required=False, type=np.int32)
args = parser.parse_args()

# Input files
GLOB_ROOT = "prods_op_mogreps-uk_20150822_03_"
EXAMPLE_GLOB = GLOB_ROOT + "??_merged.nc"
example_files = glob(EXAMPLE_GLOB)

# Other settings
percentiles = np.array([50, 90, 95, 98, 99, 99.5, 100])  # Percentiles to return
radii = np.array([30, 40, 50])
DGRID_KM = 2.2
MINUTES_PER_TIMESTEP = 5
SECONDS_PER_TIMESTEP = MINUTES_PER_TIMESTEP * 60
minutes_in_window = MINUTES_PER_TIMESTEP * args.window_length
TIME_INDEX_START = 0
TIME_INDEX_END = 12 * 21


def process_for_radius(
    radius,
    e_prec_t_max,
    member_dim,
    latitude_dim,
    longitude_dim,
):
    """
    Wrapper routine which sets up and saves the data cubes and calls post-processing
    """
    # Search radius in grid points along longitude, does not need to be integer
    rad_i = radius / DGRID_KM
    # Search radius in grid points along latitudes, does not need to be integer
    rad_j = radius / DGRID_KM
    # Calculate ensemble post-processed rainfall, obtain indices of cropped domain
    ensemble_processed_data = revamped_percentile_processing(
        e_prec_t_max, rad_i, rad_j, percentiles
    )
    percentile_dim = DimCoord(np.double(percentiles), long_name="percentile", units="1")
    make_and_save_cube(
        ensemble_processed_data,
        [percentile_dim, latitude_dim, longitude_dim],
        GLOB_ROOT + "ens_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
    )
    del ensemble_processed_data
    # Process and save individual members
    num_members = np.shape(e_prec_t_max)[0]
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
        GLOB_ROOT + "mem_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
    )
    del member_processed_data


def process_files():
    """
    The main routine of the standalone script
    Sets up the data cubes and calls the optimal time and radius-dependent processing
    """
    # Use first file to get latitudes and longitudes
    test_cube = iris.load(example_files[0])
    latitude_dim = test_cube[0].coord("grid_latitude")
    longitude_dim = test_cube[0].coord("grid_longitude")
    len_lat = test_cube[0].shape[1]
    len_lon = test_cube[0].shape[2]
    del test_cube
    # Calculate and store optimal-time rainfall
    num_members = len(example_files)
    e_prec_t_max = np.zeros((num_members, len_lat, len_lon), dtype=np.float32)
    e_t_max_index = np.zeros((num_members, len_lat, len_lon), dtype=np.uint16)
    for member in range(num_members):
        member_cube = iris.load(example_files[member])
        e_prec_t_max[member, :, :], e_t_max_index[member, :, :] = get_t_max(
            member_cube[0].data[TIME_INDEX_START:TIME_INDEX_END, :, :],
            args.window_length,
            SECONDS_PER_TIMESTEP,
        )
        del member_cube
    member_dim = DimCoord(
        np.arange(num_members, dtype=np.int32), long_name="realization", units="1"
    )
    # Add back start index when saving index cube
    make_and_save_cube(
        e_t_max_index + TIME_INDEX_START,
        [member_dim, latitude_dim, longitude_dim],
        GLOB_ROOT + "max_rain_ind_t_" + str(minutes_in_window) + ".nc",
        cube_type="index",
    )
    del e_t_max_index
    make_and_save_cube(
        e_prec_t_max,
        [member_dim, latitude_dim, longitude_dim],
        GLOB_ROOT + "max_rain_t_" + str(minutes_in_window) + ".nc",
    )
    # Post-process for each radius
    for radius in radii:
        process_for_radius(
            radius,
            e_prec_t_max,
            member_dim,
            latitude_dim,
            longitude_dim,
        )


if __name__ == "__main__":
    process_files()
