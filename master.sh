#Master script controlling regular percentile processed forecast generation

#Script requires two inputs: %Y%m%d format date string of forecast initialisation date; %H integer of forecast initialisation hour on that date (UTC).
date_info=$(date --date=$1 +"%Y%m%d")
fcst_init=$2

#KEY DIRECTORIES - CHANGE TO USER NEEDS
gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
user_dir=/home/users/bmaybee

#Retrieve and combine mogreps ensemble forecasts - requires access to MASS archive.
. ${user_dir}/mogreps_retrieval.sh $date_info $fcst_init
#datadir=${gws_dir}/mogreps_ensembles/${date_info:0:6}

#Apply percentile processing to forecasts. Note requirement for 18 member ensemble - sets failsafe $run.
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
  ssh -A sci6 "module load jaspy
  python ${user_dir}/integrated_rainfall_processing.py -f $date_info -i ${fcst_init} -nc False -loc ${gws_dir}
  python ${user_dir}/forecast_plots.py -f ${date_info} -i ${fcst_init} -loc ${gws_dir} -u ${user_dir}
  python ${user_dir}/extract_catchment_fcsts.py -f ${date_info} -i ${fcst_init} -reg EngWls -loc ${gws_dir}
  "

  # FloodForecastLookup requires older jaspy version, therefore switch to 3.7
  module load jaspy/3.7
  python ${user_dir}/FloodForecastLookup.py -f ${date_info} -i ${fcst_init} -p 98 -reg EngWls -loc ${gws_dir} -u ${user_dir}
  python ${user_dir}/present_forecasts.py -f ${date_info} -i ${fcst_init} -u ${user_dir}
  
#  . ${user_dir}/make_fcst_html.sh ${date_info}
else
  echo "No processing"
fi