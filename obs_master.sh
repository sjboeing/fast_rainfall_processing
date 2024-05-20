#!/bin/bash
#Single input: date of observations to be processed. Workflow designed for 1km UK radar network retrievals.

date_info=$(date --date=$1 +"%Y%m%d")

if [[ $(date +"%Y%m%d") == $date_info ]]; then
  date_info=$(date -d "yesterday" +"%Y%m%d")
fi
echo $date_info

#################################################################
#KEYY DIRECTORIES - CHANGE TO USER NEEDS
gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
user_dir=/home/users/bmaybee/iCASP/fast_rainfall_processing
#################################################################

#NIMROD RADAR
. ${user_dir}/radar_retrieval.sh $date_info
# Note: $radardir defined within retrieval loop. Uncomment if shell script not run.
#radardir=${gws_dir}/radar_obs/$(date -d ${date_info} +"%Y")

# Note requirement for single .nc file covering time period (calendar day by default) - sets failsafe parameter $run.
if [[ $(ls $radardir/*.nc | wc -l) == 1 ]]
  then
  rm $radardir/*.tar
  mv $radardir/*.nc $radardir/..
  rmdir $radardir
  run=1
else
  echo "Issue downloading radar files"
  run=0
fi

#Common argparse entries for python scripts: 
#   -f : %Y%m%d date of observations
#   -r : Activate radar processing rather than forecasts
#   -loc : data directory
#   -u : user directory
#   -reg : Flood forecast region; corresponds to string in name of csv tables of catchment level flood threshold values. Current options are NEng or EngWls

if [[ $run == 1 ]]; then
  # Transfer to higher memory server for processing. Essential for first routine.
  ssh -A sci6 "
  # Code designed for use on UK JASMIN collaborative platform. Python packages available in bulk via jaspy environments - full specifications can be found at https://help.jasmin.ac.uk/docs/software-on-jasmin/jaspy-envs/.
  # Require newer version of numba for post-processing algorithm.
  module load jaspy/3.10
  # ESSENTIAL - neighbourhood post-processing routine applied to raw radar rainfall fields (in units kg/m2/s!):
  python ${user_dir}/integrated_rainfall_processing.py -f ${date_info} -i 0 -r True -loc ${gws_dir}
  # forecast_plots and FloodForecastLookup both require packages within an older jaspy configuration (specifically, cartopy and ogr)
  module load jaspy/3.7
  # OPTIONAL - forecat plots made for rainfall output only; useful for detailed investigation:
  python ${user_dir}/obs_plots.py -f $date_info -loc ${gws_dir} -u ${user_dir}
  # ESSENTIAL - catchment-level look up of neighbourhood-processed rainfall field values
  python ${user_dir}/extract_catchment_fcsts.py -f ${date_info} -r True -reg EngWls -loc ${gws_dir}
  "

  # ESSENTIAL - compare catchment-level rainfall values with reference thresholds and generate flood forecast images
  python ${user_dir}/FloodForecastLookup.py -f ${date_info} -r True -reg EngWls -loc ${gws_dir} -u ${user_dir}
  # OPTIONAL - updates rainfall-based forecast plots to include radar "truth"
  # -f here is the date of the forecast initialised the day prior to observations at 18UTC. Change to meet user specifications! Only generates evaluation plots.
  python ${user_dir}/forecast_plots.py -f $(date -d "${date_info} - 1 day" +"%Y%m%d") -i 18 -e True -d $date_info -loc ${gws_dir} -u ${user_dir}
  # OPTIONAL - build pdf compiling evaluation images. Danger: hangs if ANY files missing
  #python ${user_dir}/present_evaluations.py -f ${date_info} -i 18 -u ${user_dir}
fi