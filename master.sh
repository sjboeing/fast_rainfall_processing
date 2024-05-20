#Master script controlling regular percentile processed forecast generation

#Script requires two inputs: %Y%m%d format date string of forecast initialisation date; %H integer of forecast initialisation hour on that date (UTC).
date_info=$(date --date=$1 +"%Y%m%d")
fcst_init=$2

# NOTE: running rainfall neighbourhood-processing algorithm (integrated_rainfall_processing.py) relies on numba python pacakge. Some routines also use multiprocessing parallel computation for speed, although this is not especially computationally expensive.

#################################################################
#KEY DIRECTORIES - CHANGE TO USER NEEDS
gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
user_dir=/home/users/bmaybee/iCASP/fast_rainfall_processing
#################################################################

#Retrieve and combine mogreps ensemble forecasts - requires access to MASS archive.
. ${user_dir}/mogreps_retrieval.sh $date_info $fcst_init
#datadir=${gws_dir}/mogreps_ensembles/${date_info:0:6}

# Note requirement for 18 member ensemble - sets failsafe parameter $run.
# $datadir defined within previous routine. Temporary folder for mass deposit.
if [[ $(ls $datadir/*003.pp | wc -l) == 18 ]]
then 
  mv $datadir/*.pp $datadir/..
  rmdir $datadir
  run=1
else
  echo "Incorrect number of .nc merged ensembles; problem with pull from MASS or merger script"
  run=0
fi

#Common argparse entries for python scripts: 
#   -f : %Y%m%d date of forecast initialisation
#   -i : %H integer forecast initialisation time
#   -loc : data directory
#   -u : user directory
#   -reg : Flood forecast region; corresponds to string in name of csv tables of catchment level flood threshold values. Current options are NEng or EngWls

if [[ $run == 1 ]]
then
  # Transfer to higher memory server for processing. Essential for first routine.
  ssh -A sci6 "
  # Code designed for use on UK JASMIN collaborative platform. Python packages available in bulk via jaspy environments - full specifications can be found at https://help.jasmin.ac.uk/docs/software-on-jasmin/jaspy-envs/.
  # Newer jaspy including numba environment; relied on for post-processing algorithm.
  module load jaspy/3.10
  # ESSENTIAL - neighbourhood post-processing routine applied to raw rainfall ensemble fields (in kg/m2/s!):
  python ${user_dir}/integrated_rainfall_processing.py -f $date_info -i ${fcst_init} -nc False -loc ${gws_dir}
  # forecast_plots and FloodForecastLookup both require packages within an older jaspy configuration (specifically, cartopy and ogr)
  module load jaspy/3.7
  # OPTIONAL - forecat plots made for rainfall output only; useful for detailed investigation:
  python ${user_dir}/forecast_plots.py -f ${date_info} -i ${fcst_init} -loc ${gws_dir} -u ${user_dir}
  # ESSENTIAL - catchment-level look up of neighbourhood-processed rainfall field values
  python ${user_dir}/extract_catchment_fcsts.py -f ${date_info} -i ${fcst_init} -reg EngWls -loc ${gws_dir}
  "
  
  # ESSENTIAL - compare catchment-level rainfall values with reference thresholds and generate flood forecast images
  python ${user_dir}/FloodForecastLookup.py -f ${date_info} -i ${fcst_init} -p 98 -reg EngWls -loc ${gws_dir} -u ${user_dir}
  # OPTIONAL - build pdf compiling flood and rainfall forecast images. Danger: hangs if ANY files missing
  #python ${user_dir}/present_forecasts.py -f ${date_info} -i ${fcst_init} -u ${user_dir}
  
else
  echo "No processing"
fi