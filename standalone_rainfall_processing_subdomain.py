from numba import njit, jit, prange, set_num_threads
from numba.types import bool_
import numpy as np
from glob import glob
import argparse
import iris
from iris.coords import DimCoord

# Argument parser
parser = argparse.ArgumentParser(
    prog="Percentile processor",
    description="A numba implementation of fast neighbourhood percentile processing",
)

parser.add_argument("-w", "--window_length", default=12, required=False, type=np.int32)
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
    This bit of code gets the optimal-time precipitation and the corresponding
    time index. It expects a 3D (time,lat,lon) input array, i.e one member at a
    time, which is more memory efficient.
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
    # Index of optimal times
    t_max_index = np.zeros((len_i, len_j), dtype=np.uint16)
    # Corresponding precipitation
    prec_t_max = np.zeros((len_i, len_j), dtype=np.float32)
    # Find the right time window for each latitude and longitude
    # Parallelise over longitude
    for i_index in prange(len_i):
        for j_index in range(len_j):
            # Sum over the initial window length
            prec_t = (
                np.sum(input_data[:window_length, i_index, j_index])
                * seconds_per_timestep
            )
            prec_t_max[i_index, j_index] = prec_t
            # Running window
            # Remove a time step and add a time step to a running total
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


# Sorting using subdomain strategy
# That **should** scale quite well for large radii
# And handles the corners correctly, hopefully
# Only make a complete new sort domain every [i_skip,j_skip] points in i and j
# Remove indices which are no longer relevant to the current subdomain from in_sort array
# The additional functions should make it a little easier to understand the resulting code

# Use a backfill approach
@njit
def filter_values_for_subdomain(
    in_sort_values,
    in_sort_e,
    in_sort_i,
    in_sort_j,
    subdomain_i_start,
    subdomain_j_start,
):
    keeper_index = 0
    len_values = len(in_sort_values)
    for in_sort_index in range(len_values):
        if in_sort_j[in_sort_index] >= subdomain_j_start:
            # Check below should always be satisfied in current configuration
            # But keep it for now, since it helps generalise
            # Do assume i and j are always increasing though
            # So only check for values LOWER than start index
            if in_sort_i[in_sort_index] >= subdomain_i_start:
                in_sort_values[keeper_index] = in_sort_values[in_sort_index]
                in_sort_e[keeper_index] = in_sort_e[in_sort_index]
                in_sort_i[keeper_index] = in_sort_i[in_sort_index]
                in_sort_j[keeper_index] = in_sort_j[in_sort_index]
                keeper_index = keeper_index + 1
    in_sort_values = in_sort_values[:keeper_index]
    in_sort_e = in_sort_e[:keeper_index]
    in_sort_i = in_sort_i[:keeper_index]
    in_sort_j = in_sort_j[:keeper_index]
    return in_sort_values, in_sort_e, in_sort_i, in_sort_j


# Add new indices
@njit
def add_new_values_to_subdomain(e_prec_t_max, subdomain_i_indices, subdomain_j_indices):
    len_e = np.shape(e_prec_t_max)[0]  # Number of members
    added_len = len_e * len(subdomain_i_indices) * len(subdomain_j_indices)
    added_values = np.zeros(added_len, dtype=np.float32)
    added_e = np.zeros(added_len, dtype=np.uint8)
    added_i = np.zeros(added_len, dtype=np.uint16)
    added_j = np.zeros(added_len, dtype=np.uint16)
    added_index = 0
    for e_index in range(len_e):
        for i_index in subdomain_i_indices:
            for j_index in subdomain_j_indices:
                added_values[added_index] = e_prec_t_max[e_index, i_index, j_index]
                added_e[added_index] = e_index
                added_i[added_index] = i_index
                added_j[added_index] = j_index
                added_index = added_index + 1
    indices_for_add_to_sort = np.argsort(added_values)
    sorted_a_values = added_values.take(indices_for_add_to_sort)
    sorted_a_e = added_e.take(indices_for_add_to_sort)
    sorted_a_i = added_i.take(indices_for_add_to_sort)
    sorted_a_j = added_j.take(indices_for_add_to_sort)
    # Sort the added values
    return sorted_a_values, sorted_a_e, sorted_a_i, sorted_a_j


@njit
def combine_sorted_arrays(
    in_sort_values,
    in_sort_e,
    in_sort_i,
    in_sort_j,
    added_values,
    added_e,
    added_i,
    added_j,
):
    keeper_index = len(in_sort_values)
    add_to_sort_index = len(added_values)
    new_in_sort_len = keeper_index + add_to_sort_index
    # Assumption: both input arrays are already sorted
    new_in_sort_values = np.zeros(new_in_sort_len, dtype=np.float32)
    new_in_sort_e = np.zeros(new_in_sort_len, dtype=np.uint8)
    new_in_sort_i = np.zeros(new_in_sort_len, dtype=np.uint16)
    new_in_sort_j = np.zeros(new_in_sort_len, dtype=np.uint16)
    is_index = 0
    as_index = 0
    for nis_index in range(new_in_sort_len):
        if (is_index < keeper_index) and (as_index < add_to_sort_index):
            if in_sort_values[is_index] < added_values[as_index]:
                new_in_sort_values[nis_index] = in_sort_values[is_index]
                new_in_sort_e[nis_index] = in_sort_e[is_index]
                new_in_sort_i[nis_index] = in_sort_i[is_index]
                new_in_sort_j[nis_index] = in_sort_j[is_index]
                is_index = is_index + 1
            else:
                new_in_sort_values[nis_index] = added_values[as_index]
                new_in_sort_e[nis_index] = added_e[as_index]
                new_in_sort_i[nis_index] = added_i[as_index]
                new_in_sort_j[nis_index] = added_j[as_index]
                as_index = as_index + 1
        elif is_index < keeper_index:
            new_in_sort_values[nis_index] = in_sort_values[is_index]
            new_in_sort_e[nis_index] = in_sort_e[is_index]
            new_in_sort_i[nis_index] = in_sort_i[is_index]
            new_in_sort_j[nis_index] = in_sort_j[is_index]
            is_index = is_index + 1
        elif as_index < add_to_sort_index:
            new_in_sort_values[nis_index] = added_values[as_index]
            new_in_sort_e[nis_index] = added_e[as_index]
            new_in_sort_i[nis_index] = added_i[as_index]
            new_in_sort_j[nis_index] = added_j[as_index]
            as_index = as_index + 1
    # Replace full in_sort list
    return new_in_sort_values, new_in_sort_e, new_in_sort_i, new_in_sort_j


@njit
def extract_percentiles_for_location(
    in_sort_values,
    in_sort_i,
    in_sort_j,
    loc_i,
    loc_j,
    rad_i,
    rad_j,
    percentiles,
):
    # Run through the in_sort list twice, once to extract the number of hits, and once to extract the corresponding percentiles
    # This means there is no need to store the sorted indices
    # Going through the list twice will be needed to generalise to topographic filters and deal with boundaries.
    i_rad_i2 = 1.0 / (rad_i * rad_i)
    i_rad_j2 = 1.0 / (rad_j * rad_j)
    n_valid_indices = 0
    len_values = np.shape(in_sort_values)[0]
    for is_index in range(len_values):
        shift_i = in_sort_i[is_index] - loc_i
        shift_j = in_sort_j[is_index] - loc_j
        if (shift_j * shift_j) * i_rad_j2 + (shift_i * shift_i) * i_rad_i2 <= 1.0:
            n_valid_indices = n_valid_indices + 1
    len_p = np.shape(percentiles)[0]
    percentile_rankings = np.zeros(len_p, dtype=np.uint32)
    for p_index in range(len_p):
        percentile_rankings[p_index] = round(
            (n_valid_indices - 1) * percentiles[p_index] / 100.0
        )
    percentile_ranking_index = 0
    percentile_values = np.zeros(len_p, dtype=np.float32)
    in_sort_index = 0
    valid_index = 0
    while valid_index < n_valid_indices:
        shift_i = in_sort_i[in_sort_index] - loc_i
        shift_j = in_sort_j[in_sort_index] - loc_j
        if (shift_j * shift_j) * i_rad_j2 + (shift_i * shift_i) * i_rad_i2 <= 1.0:
            if valid_index == percentile_rankings[percentile_ranking_index]:
                percentile_values[percentile_ranking_index] = in_sort_values[
                    in_sort_index
                ]
                percentile_ranking_index = percentile_ranking_index + 1
            else:
                in_sort_index = in_sort_index + 1
                valid_index = valid_index + 1
        else:
            in_sort_index = in_sort_index + 1
    return percentile_values


@njit
def get_indices_of_subdomain(
    subdomain_index_start, skip_index, int_rad_index, len_index
):
    subdomain_index_end = min(
        subdomain_index_start + skip_index + 2 * int_rad_index, len_index
    )
    return np.arange(subdomain_index_start, subdomain_index_end)


@njit
def get_indices_to_evaluate(
    subdomain_index_start, skip_index, int_rad_index, len_index
):
    subdomain_index_end = min(
        subdomain_index_start + skip_index + 2 * int_rad_index, len_index
    )
    if subdomain_index_start == 0:
        index_to_start_eval = subdomain_index_start
    else:
        index_to_start_eval = subdomain_index_start + int_rad_index
    if subdomain_index_end == len_index:
        index_to_end_eval = len_index
    else:
        index_to_end_eval = subdomain_index_start + skip_index + int_rad_index
    return np.arange(index_to_start_eval, index_to_end_eval)


@njit(parallel=True)
def revamped_percentile_processing(
    e_prec_t_max, rad_i, rad_j, percentiles, i_skip_subdomain=10, j_skip_subdomain=10
):
    int_rad_i = int(np.ceil(rad_i))
    int_rad_j = int(np.ceil(rad_j))
    len_e = e_prec_t_max.shape[0]  # Number of ensemble members
    len_i = e_prec_t_max.shape[1]  # Number of lat indices
    len_j = e_prec_t_max.shape[2]  # Number of lon indices
    len_p = len(percentiles)
    processed_data = np.zeros((len_p, len_i, len_j), dtype=np.float32)
    # Handle subdomain i start index in parallel
    i_start_indices = np.arange(0, len_i, i_skip_subdomain)
    len_isi = len(i_start_indices)
    for isi_index in prange(len_isi):
        subdomain_i_start = i_start_indices[isi_index]
        is_subdomain = get_indices_of_subdomain(
            subdomain_i_start, i_skip_subdomain, int_rad_i, len_i
        )
        is_to_evaluate = get_indices_to_evaluate(
            subdomain_i_start, i_skip_subdomain, int_rad_i, len_i
        )
        in_sort_values = np.array([0.0], dtype=np.float32)
        in_sort_e = np.array([0], dtype=np.uint8)
        in_sort_i = np.array([-1], dtype=np.uint16)
        in_sort_j = np.array([-1], dtype=np.uint16)
        sub_domain_prev_j_end = -1
        for subdomain_j_start in range(0, len_j, j_skip_subdomain):
            js_subdomain = get_indices_of_subdomain(
                subdomain_j_start, j_skip_subdomain, int_rad_j, len_j
            )
            new_js_subdomain = js_subdomain[js_subdomain[:] > sub_domain_prev_j_end]
            sub_domain_prev_j_end = js_subdomain[-1]
            js_to_evaluate = get_indices_to_evaluate(
                subdomain_j_start, j_skip_subdomain, int_rad_j, len_j
            )
            # Remove values no longer needed
            (
                in_sort_values,
                in_sort_e,
                in_sort_i,
                in_sort_j,
            ) = filter_values_for_subdomain(
                in_sort_values,
                in_sort_e,
                in_sort_i,
                in_sort_j,
                subdomain_i_start,
                subdomain_j_start,
            )
            (
                sorted_a_values,
                sorted_a_e,
                sorted_a_i,
                sorted_a_j,
            ) = add_new_values_to_subdomain(
                e_prec_t_max, is_subdomain, new_js_subdomain
            )
            in_sort_values, in_sort_e, in_sort_i, in_sort_j = combine_sorted_arrays(
                in_sort_values,
                in_sort_e,
                in_sort_i,
                in_sort_j,
                sorted_a_values,
                sorted_a_e,
                sorted_a_i,
                sorted_a_j,
            )
            # ~ # Now go add the new indices
            for loc_i in is_to_evaluate:
                for loc_j in js_to_evaluate:
                    percentile_values = extract_percentiles_for_location(
                        in_sort_values,
                        in_sort_i,
                        in_sort_j,
                        loc_i,
                        loc_j,
                        rad_i,
                        rad_j,
                        percentiles,
                    )
                    processed_data[:, loc_i, loc_j] = percentile_values
    return processed_data


def make_and_save_cube(data, dims, cube_filename, cube_type="amount"):
    """
    Routine to set up and save a cube with given dimensions, which adds the
    relevant attributes
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
        raise ValueError("cube data to create has unexpected shape")
    cube.attributes["source"] = "revamped_rainfall_postprocessing on MOGREPS_UK data"
    if cube_type == "amount":
        cube.long_name = "Rainfall amount"
        cube.var_name = "rainfall_amount"
        cube.units = "mm"
        iris.fileformats.netcdf.save(
            cube,
            cube_filename,
            netcdf_format="NETCDF4",
            zlib=True,
            complevel=4,
            least_significant_digit=2,
        )
    elif cube_type == "index":
        cube.long_name = "Start index within full time series"
        cube.var_name = "start_index"
        cube.units = "1"
        iris.fileformats.netcdf.save(
            cube,
            cube_filename,
            netcdf_format="NETCDF4",
            zlib=True,
            complevel=4,
        )
    else:
        raise ValueError("cube to export has unexpected cube_type")
    del cube


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
    rad_i = radius / dgrid_km
    # Search radius in grid points along latitudes, does not need to be integer
    rad_j = radius / dgrid_km
    # Calculate ensemble post-processed rainfall, obtain indices of cropped domain
    ensemble_processed_data = revamped_percentile_processing(
        e_prec_t_max, rad_i, rad_j, percentiles
    )
    # Extract latitudes and longitudes to save
    percentile_dim = DimCoord(np.double(percentiles), long_name="percentile", units="1")
    make_and_save_cube(
        ensemble_processed_data,
        [percentile_dim, latitude_dim, longitude_dim],
        glob_root + "ens_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
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
        glob_root + "mem_pp_r" + str(radius) + "_t_" + str(minutes_in_window) + ".nc",
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
            member_cube[0].data[time_index_start:time_index_end, :, :],
            args.window_length,
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
        glob_root + "max_rain_ind_t_" + str(minutes_in_window) + ".nc",
        cube_type="index",
    )
    del e_t_max_index
    make_and_save_cube(
        e_prec_t_max,
        [member_dim, latitude_dim, longitude_dim],
        glob_root + "max_rain_t_" + str(minutes_in_window) + ".nc",
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
