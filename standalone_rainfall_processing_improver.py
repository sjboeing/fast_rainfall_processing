"""This module contains a stand-alone version of the postprocessing algorithm."""
from glob import glob
import argparse
import numpy as np
import iris
from iris.coords import DimCoord
from improver.nbhood.nbhood import GeneratePercentilesFromANeighbourhood
from utils import get_t_max

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


def process_files():
    """
    The main routine of the standalone script
    Sets up the data cubes and calls the optimal time and radius-dependent processing
    """
    # Use first file to get latitudes and longitudes
    test_cube = iris.load(example_files[0])
    len_lat = test_cube[0].shape[1]
    len_lon = test_cube[0].shape[2]
    latitude_m_dim = DimCoord(
        1000.0 * DGRID_KM * np.arange(len_lat),
        standard_name="projection_x_coordinate",
        units="m",
    )
    longitude_m_dim = DimCoord(
        1000.0 * DGRID_KM * np.arange(len_lon),
        standard_name="projection_y_coordinate",
        units="m",
    )
    latitude_m_dim.axis = "X"
    longitude_m_dim.axis = "Y"
    del test_cube
    # Calculate and store optimal-time rainfall
    num_members = len(example_files)
    e_prec_t_max = np.zeros((num_members, len_lat, len_lon), dtype=np.float32)
    e_t_max_index = np.zeros((num_members, len_lat, len_lon), dtype=np.int32)
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
    cube = iris.cube.Cube(
        e_prec_t_max,
        dim_coords_and_dims=[
            (member_dim, 0),
            (latitude_m_dim, 1),
            (longitude_m_dim, 2),
        ],
    )
    cube.long_name = "Rainfall amount"
    cube.var_name = "rainfall_amount"
    cube.units = "mm"
    for radius in radii:
        # noticed this may need another argument "neighbourhood_method" for some versions of improver
        cubeout = GeneratePercentilesFromANeighbourhood(
            radii=1000.0 * radius,
            percentiles=percentiles,
        )(cube)
        iris.save(
            cubeout,
            "improver_"
            + GLOB_ROOT
            + "mem_pp_r"
            + str(radius)
            + "_t_"
            + str(minutes_in_window)
            + ".nc",
        )


if __name__ == "__main__":
    process_files()
