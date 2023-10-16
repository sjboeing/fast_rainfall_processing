"""This module contains the functions for the numba-based post-processing."""
from numba import njit, prange, set_num_threads
import numpy as np
import iris

# Number of threads to use
NTHREADS = 4
set_num_threads(NTHREADS)


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
    """
    Removes points outside the subdomain from the sorted arrays
    """
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
    """
    Adds new points in subdomain to new sorted arrays
    """
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
    # Sort the added values
    indices_for_add_to_sort = np.argsort(added_values)
    sorted_a_values = added_values.take(indices_for_add_to_sort)
    sorted_a_e = added_e.take(indices_for_add_to_sort)
    sorted_a_i = added_i.take(indices_for_add_to_sort)
    sorted_a_j = added_j.take(indices_for_add_to_sort)
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
    """
    Adds the new sorted array values to existing (filtered) sorting array
    """
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
    """
    Obtains the required percentile values for a single point in the domain.
    Run through the in_sort list twice, once to extract the number of hits, and once to extract the corresponding percentiles
    This means there is no need to store the sorted indices
    Going through the list twice will be needed to generalise to topographic filters and deal with boundaries.
    """
    i_rad_i2 = 1.0 / (rad_i * rad_i)
    i_rad_j2 = 1.0 / (rad_j * rad_j)
    mask = ((in_sort_j[:] - loc_j) * (in_sort_j[:] - loc_j)) * i_rad_j2 + (
        (in_sort_i[:] - loc_i) * (in_sort_i[:] - loc_i)
    ) * i_rad_i2 <= 1.0
    selected_values = in_sort_values[mask]
    n_valid_indices = len(selected_values)
    len_p = np.shape(percentiles)[0]
    percentile_values = np.zeros(len_p, dtype=np.float32)
    for p_index in range(len_p):
        percentile_ranking = round((n_valid_indices - 1) * percentiles[p_index] / 100.0)
        percentile_values[p_index] = selected_values[percentile_ranking]
    return percentile_values


@njit
def get_indices_of_subdomain(
    subdomain_index_start, skip_index, int_rad_index, len_index
):
    """
    Gets the full extent used for this subdomain (including points not evaluated)
    along one axis
    """
    subdomain_index_end = min(
        subdomain_index_start + skip_index + 2 * int_rad_index, len_index
    )
    return np.arange(subdomain_index_start, subdomain_index_end)


@njit
def get_indices_to_evaluate(
    subdomain_index_start, skip_index, int_rad_index, len_index
):
    """
    Gets the indices where evaluation is done for this subdomain
    along one axis
    """
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
    """
    A revised implementation of the neighbourhood post-processing.
    This version splits the domain into "subdomains", the values
    are first sorted for each subdomain, and the relevant indices
    are then selected.
    This **should** scale quite well for large radii
    and handles the corners correctly, hopefully
    Only make a complete new sort domain every [i_skip_subdomain,j_skip_subdomain] points in i and j
    The additional functions should make it a little easier to understand the resulting code
    """
    int_rad_i = int(np.ceil(rad_i))
    int_rad_j = int(np.ceil(rad_j))
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
