#!/bin/bash
#Single input: date of observations to be processed. Workflow designed for 1km UK radar network retrievals.

date_info=$(date --date=$1 +"%Y%m%d")

if [[ $(date +"%Y%m%d") == $date_info ]]; then
  date_info=$(date -d "yesterday" +"%Y%m%d")
fi
echo $date_info

#KEYY DIRECTORIES - CHANGE TO USER NEEDS
gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
user_dir=/home/users/bmaybee

#NIMROD RADAR
. ${user_dir}/radar_retrieval.sh $date_info
# $radardir defined within retrieval loop
#radardir=${gws_dir}/radar_obs/$(date -d ${date_info} +"%Y")/temp
#radardir=${gws_dir}/radar_obs/temp

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
  ssh -A sci6 "
  module load jaspy
  python ${user_dir}/integrated_rainfall_processing.py -f ${date_info} -i 0 -r True -loc ${gws_dir}
  python ${user_dir}/obs_plots.py -f $date_info -loc ${gws_dir} -u ${user_dir}
  python ${user_dir}/extract_catchment_fcsts.py -f ${date_info}_00 -r True -reg EngWls -loc ${gws_dir}
  "
  # FloodForecastLookup requires older jaspy version, therefore switch to 3.7
  module load jaspy/3.7
  python ${user_dir}/FloodForecastLookup.py -f ${date_info} -r True -reg EngWls -loc ${gws_dir} -u ${user_dir}
  # -f here is the date of the forecast initialised the day prior to observations at 18UTC. Change to meet user specifications! Only generates evaluation plots.
  python ${user_dir}/forecast_plots.py -f $(date -d "${date_info} - 1 day" +"%Y%m%d") -i 18 -e True -d $date_info -loc ${gws_dir} -u ${user_dir}
  python ${user_dir}/present_evaluations.py -f ${date_info} -i 18 -loc ${gws_dir} -u ${user_dir}
  #. ${user_dir}/make_obs_html.sh $date_info
fi