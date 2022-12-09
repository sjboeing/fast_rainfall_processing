# -*- coding: utf-8 -*-
from numba import njit, prange, set_num_threads
from numba.types import bool_
import numpy as np
from glob import glob
import argparse
import iris
from iris.coords import DimCoord

# Argument parser
parser = argparse.ArgumentParser(
    prog="Percentile processor",
    description="A numba implementation of fast neighboorhood percentile processing",
)

parser.add_argument("-w", "--window_length", default=12, required=False, type=np.int32)
parser.add_argument(
    "-s", "--stride_ij", default=1, required=False, type=np.int32
)  # How many points to skip in the post-processing
args = parser.parse_args()

# Input files
glob_root = "prods_op_mogreps-uk_20150822_03_"
example_glob = glob_root + "??_merged.nc"
example_files = glob(example_glob)

# Number of threads to use
nthreads = 4
set_num_threads(nthreads)

# Other settings
percentiles = np.array([50, 90, 95, 98, 99, 99.5, 100])  # Percentiles to return
radii = np.array([30, 40, 50])
dgrid_km = 2.2
minutes_per_timestep = 5
seconds_per_timestep = minutes_per_timestep * 60
minutes_in_window = minutes_per_timestep * args.window_length
time_index_start = 0
time_index_end = 12 * 21


@njit(parallel=True)
def get_t_max(input_data, window_length, seconds_per_timestep):
    """
    This bit of code gets the optimal-time precipitation and the corresponding time index
    It expects a 3D (time,lat,lon) input array, i.e one member at a time, which is more memory efficient
    """
    if input_data.ndim != 3:
        raise ValueError("get_t_max: input has wrong shape")
    len_t = input_data.shape[0]  # Number of time steps
    len_i = input_data.shape[1]  # Number of lat indices
    len_j = input_data.shape[2]  # Number of lon indices
    if window_length > len_t:
        raise ValueError(
            "get_t_max: window_length larger than number of time steps provided"
        )
    t_max_index = np.zeros((len_i, len_j), dtype=np.uint16)  # Index of optimal times
    prec_t_max = np.zeros(
        (len_i, len_j), dtype=np.float32
    )  # Corresponding precipitation
    # Find the right time window for each member, latitude and longitude
    for i_index in prange(len_i):  # Parallelise along longitude
        for j_index in range(len_j):
            # Sum over the initial window length
            prec_t = (
                np.sum(input_data[:window_length, i_index, j_index])
                * seconds_per_timestep
            )
            prec_t_max[i_index, j_index] = prec_t
            # Running window: remove a time step and add a time step to running total
            for t_index in range(len_t - window_length):
                prec_t = (
                    prec_t
                    + (
                        input_data[t_index + window_length, i_index, j_index]
                        - input_data[t_index, i_index, j_index]
                    )
                    * seconds_per_timestep
                )
                if prec_t > prec_t_max[i_index, j_index]:
                    prec_t_max[i_index, j_index] = prec_t
                    t_max_index[i_index, j_index] = t_index
    return prec_t_max, t_max_index


@njit(parallel=True)
def fast_percentile_processing(e_prec_t_max, rad_i, rad_j, percentiles, stride_ij=1):
    """
    Percentile search algorithm, rewritten in numba
    - Expects a 3D array (member,lat,lon)
    - No topographic or land/sea masking (would be much easier to add in this version
      one idea would be to ignore data where the point is more than 200m higher)
    - Returns cropped output, only where a full search circle is available
    """
    if e_prec_t_max.ndim != 3:
        raise ValueError("e_prec_t_max.ndim: input has wrong shape")
    for percentile in percentiles:
        if percentile < 0 or percentile > 100:
            raise ValueError("e_prec_t_max.ndim: invalid percentile")
    int_rad_i = int(np.ceil(rad_i))
    int_rad_j = int(np.ceil(rad_j))
    len_e = e_prec_t_max.shape[0]  # Number of ensemble members
    len_i = e_prec_t_max.shape[1]  # Number of lat indices
    len_j = e_prec_t_max.shape[2]  # Number of lon indices
    len_p = len(percentiles)
    # Set up output array
    # Crop to only produce output where a full search circle is available
    # Store indices where to evaluate neighbourhood postprocessing
    # Ensure same indices always selected for evaluation, independent of radius
    prelim_range_i_pp = np.arange(0, len_i, stride_ij)
    prelim_range_j_pp = np.arange(0, len_j, stride_ij)
    filter_range_i_pp = (prelim_range_i_pp >= int_rad_i) & (
        prelim_range_i_pp < len_i - int_rad_i
    )
    filter_range_j_pp = (prelim_range_j_pp >= int_rad_j) & (
        prelim_range_j_pp < len_j - int_rad_j
    )
    range_i_pp = prelim_range_i_pp[filter_range_i_pp]
    range_j_pp = prelim_range_j_pp[filter_range_j_pp]
    len_i_pp = len(range_i_pp)
    len_j_pp = len(range_j_pp)
    if len_i_pp == 0 or len_j_pp == 0:
        raise ValueError("e_prec_t_max.ndim: no valid output points")
    processed_data = np.zeros((len_p, len_i_pp, len_j_pp), dtype=np.float32)
    # Dimensions of search window for post-processing
    len_i_search = 2 * int_rad_i + 1
    len_j_search = 2 * int_rad_j + 1
    # Number of points checked in post-processing
    len_eij_search = len_e * len_i_search * len_j_search
    # In this case, the activate_point filter is always the same, but this could change in future
    # for example, if we want to include topographic or land/sea processing
    # or use a radius which uses exact great circle distances
    activate_point = np.full((len_i_search, len_j_search), False, dtype=bool_)
    for i_search in range(len_i_search):
        for j_search in range(len_j_search):
            rad_i_scale = (1.0 * (i_search - rad_i)) / rad_i
            rad_j_scale = (1.0 * (j_search - rad_j)) / rad_j
            if rad_i_scale * rad_i_scale + rad_j_scale * rad_j_scale <= 1.0:
                activate_point[i_search, j_search] = True
    for i_mapping in prange(len_i_pp):  # Parallel over longitude
        # Longitude index of point where post_processing is done
        i_index = range_i_pp[i_mapping]
        # Arrays to keep track of values currently in sort
        in_sort_values = np.zeros(0, dtype=np.float32)
        in_sort_e = np.zeros(0, dtype=np.uint8)
        in_sort_i = np.zeros(0, dtype=np.uint16)
        in_sort_j = np.zeros(0, dtype=np.uint16)
        # Arrays to keep track of values to add to sorted list
        add_to_sort_values = np.zeros(len_eij_search, dtype=np.float32)
        add_to_sort_e = np.zeros(len_eij_search, dtype=np.uint8)
        add_to_sort_i = np.zeros(len_eij_search, dtype=np.uint16)
        add_to_sort_j = np.zeros(len_eij_search, dtype=np.uint16)
        for j_mapping in range(len_j_pp):
            # Longitude index of point where post_processing is done
            j_index = range_j_pp[j_mapping]
            # Go over the present sort list and remove entries that are no longer relevant
            # Mark retained points
            retain_point = np.full((len_i_search, len_j_search), False, dtype=bool_)
            backfill_index = 0
            for retained_index in range(len(in_sort_values)):
                # Check if cell in existing list is still active
                # Note that list should already be sorted
                delta_i = in_sort_i[retained_index] - i_index
                delta_j = in_sort_j[retained_index] - j_index
                rad_i_scale = (1.0 * delta_i) / rad_i
                rad_j_scale = (1.0 * delta_j) / rad_j
                if (rad_i_scale * rad_i_scale + rad_j_scale * rad_j_scale) > 1.0:
                    continue
                retain_point[delta_i + int_rad_i, delta_j + int_rad_j] = True
                # keep value in sorted list using backfill
                in_sort_values[backfill_index] = in_sort_values[retained_index]
                in_sort_e[backfill_index] = in_sort_e[retained_index]
                in_sort_i[backfill_index] = in_sort_i[retained_index]
                in_sort_j[backfill_index] = in_sort_j[retained_index]
                backfill_index = backfill_index + 1
            # Check for values that need to be added
            # Index counter
            add_to_sort_index = 0
            for e_target in range(len_e):
                for i_search in range(len_i_search):
                    i_target = i_index + i_search - int_rad_i
                    for j_search in range(len_j_search):
                        j_target = j_index + j_search - int_rad_j
                        a_point = activate_point[i_search, j_search]
                        r_point = retain_point[i_search, j_search]
                        if r_point or not a_point:
                            continue
                        add_to_sort_values[add_to_sort_index] = e_prec_t_max[
                            e_target, i_target, j_target
                        ]
                        add_to_sort_e[add_to_sort_index] = e_target
                        add_to_sort_i[add_to_sort_index] = i_target
                        add_to_sort_j[add_to_sort_index] = j_target
                        add_to_sort_index = add_to_sort_index + 1
            # Sort the values that need adding first, use argsort to get index order
            indices_for_add_to_sort = np.argsort(add_to_sort_values[:add_to_sort_index])
            add_to_sort_values = add_to_sort_values.take(indices_for_add_to_sort)
            add_to_sort_e = add_to_sort_e.take(indices_for_add_to_sort)
            add_to_sort_i = add_to_sort_i.take(indices_for_add_to_sort)
            add_to_sort_j = add_to_sort_j.take(indices_for_add_to_sort)
            # Combine with retained values
            in_sort_values = np.concatenate(
                (in_sort_values[:backfill_index], add_to_sort_values)
            )
            in_sort_e = np.concatenate((in_sort_e[:backfill_index], add_to_sort_e))
            in_sort_i = np.concatenate((in_sort_i[:backfill_index], add_to_sort_i))
            in_sort_j = np.concatenate((in_sort_j[:backfill_index], add_to_sort_j))
            # Sort combined list with a mergesort (should be efficient for presorted lists)
            indices_for_in_sort = np.argsort(in_sort_values, kind="mergesort")
            in_sort_values = in_sort_values.take(indices_for_in_sort)
            in_sort_e = in_sort_e.take(indices_for_in_sort)
            in_sort_i = in_sort_i.take(indices_for_in_sort)
            in_sort_j = in_sort_j.take(indices_for_in_sort)
            # Combine the relevant data
            for p_mapping in range(len_p):
                p_index = round(
                    (len(in_sort_values) - 1) * percentiles[p_mapping] / 100.0
                )
                processed_data[p_mapping, i_mapping, j_mapping] = in_sort_values[
                    p_index
                ]
    return processed_data, range_i_pp, range_j_pp


def make_cube(data, dims, cube_type="amount"):
    """
    Routine to set up a cube with given dimensions,
    which adds the relevant attributes
    """
    if len(dims) == 3:
        cube = iris.cube.Cube(
            data, dim_coords_and_dims=[(dims[0], 0), (dims[1], 1), (dims[2], 2)]
        )
    elif len(dims) == 4:
        cube = iris.cube.Cube(
            data,
            dim_coords_and_dims=[
                (dims[0], 0),
                (dims[1], 1),
                (dims[2], 2),
                (dims[3], 3),
            ],
        )
    else:
        raise ValueError("cube to export has unexpected shape")
    if cube_type == "amount":
        cube.long_name = "Rainfall amount"
        cube.var_name = "rainfall_amount"
        cube.standard_name = "rainfall_amount"
        cube.units = "mm"
    elif cube_type == "index":
        cube.long_name = "Start index within full time series"
        cube.var_name = "start_index"
        cube.units = "1"
    else:
        raise ValueError("cube to export has unexpected cube_type")
    return cube


def save_cube(cube, cube_filename, cube_type="amount"):
    """
    Routine that saves to NetCDF with compression
    """
    if cube_type == "amount":
        iris.fileformats.netcdf.save(
            cube,
            cube_filename,
            netcdf_format="NETCDF4",
            zlib=True,
            complevel=4,
            least_significant_digit=2,
        )
    elif cube_type == "index":
        iris.fileformats.netcdf.save(
            cube,
            cube_filename,
            netcdf_format="NETCDF4",
            zlib=True,
            complevel=4,
        )
    else:
        raise ValueError("cube to export has unexpected cube_type")


def process_for_radius(
    radius,
    e_prec_t_max,
    stride_ij,
    num_members,
    member_dim,
    latitude_dim,
    longitude_dim,
):
    """
    Wrapper routine which sets up and saves the data cubes
    """
    rad_i = (
        radius / dgrid_km
    )  # Search radius in grid points along longitude, does not need to be integer
    rad_j = (
        radius / dgrid_km
    )  # Search radius in grid points along latitudes, does not need to be integer
    # Calculate post-processed rainfall, obtain indices of cropped domain
    ensemble_processed_data, range_i_pp, range_j_pp = fast_percentile_processing(
        e_prec_t_max, rad_i, rad_j, percentiles, stride_ij
    )
    reduced_latitude_dim = latitude_dim[range_i_pp]
    reduced_longitude_dim = longitude_dim[range_j_pp]
    percentile_dim = DimCoord(np.double(percentiles), long_name="percentile", units="1")
    ensemble_processed_data_cube = make_cube(
        ensemble_processed_data,
        [percentile_dim, reduced_latitude_dim, reduced_longitude_dim],
    )
    len_reduced_lat = ensemble_processed_data_cube.shape[1]
    len_reduced_lon = ensemble_processed_data_cube.shape[2]
    save_cube(
        ensemble_processed_data_cube,
        glob_root + "ens_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
    )
    del ensemble_processed_data, ensemble_processed_data_cube
    member_processed_data = np.zeros(
        (
            num_members,
            len(percentiles),
            len_reduced_lat,
            len_reduced_lon,
        ),
        dtype=np.float32,
    )
    for member in range(num_members):
        (
            member_processed_data[member],
            range_i_pp,
            range_j_pp,
        ) = fast_percentile_processing(
            e_prec_t_max[member, :, :][None, :, :], rad_i, rad_j, percentiles, stride_ij
        )
    member_processed_data_cube = make_cube(
        member_processed_data,
        [member_dim, percentile_dim, reduced_latitude_dim, reduced_longitude_dim],
    )
    save_cube(
        member_processed_data_cube,
        glob_root + "mem_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
    )
    del member_processed_data, member_processed_data_cube


def process_files():
    """
    The main routine of the standalone script
    Sets up the data cubes and calls the optimal time
    And radius-dependent processing
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
            member_cube[0].data[time_index_start:time_index_end, :, :],
            args.window_length,
            seconds_per_timestep,
        )
        del member_cube
    member_dim = DimCoord(
        np.arange(num_members, dtype=np.int32), long_name="member", units="1"
    )
    # Add back start index when saving index cube
    e_t_max_index_cube = make_cube(
        e_t_max_index + time_index_start,
        [member_dim, latitude_dim, longitude_dim],
        cube_type="index",
    )
    save_cube(
        e_t_max_index_cube,
        glob_root + "max_rain_ind_t_" + str(minutes_in_window) + ".nc",
        cube_type="index",
    )
    del e_t_max_index, e_t_max_index_cube
    e_prec_t_max_cube = make_cube(
        e_prec_t_max, [member_dim, latitude_dim, longitude_dim]
    )
    save_cube(
        e_prec_t_max_cube, glob_root + "max_rain_t_" + str(minutes_in_window) + ".nc"
    )
    del e_prec_t_max_cube
    # Post-process for each radius
    stride_ij = args.stride_ij
    for radius in radii:
        process_for_radius(
            radius,
            e_prec_t_max,
            stride_ij,
            num_members,
            member_dim,
            latitude_dim,
            longitude_dim,
        )


if __name__ == "__main__":
    process_files()
