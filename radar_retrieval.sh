#!/bin/bash
# PREREQ's : access to MASS Nimrod radar files (requires Met Office approval)

date_info=$1
gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
radardir=${gws_dir}/radar_obs/$(date -d ${date_info} +"%Y")/temp_dir
#radardir=${gws_dir}/radar_obs/temp
user_dir=/home/users/bmaybee/MO_testbed_2023
mkdir ${radardir}
rm $radardir/*

day=$(date -d ${date_info} +"%Y%m%d")
day1=$(date -d ${date_info}" +1 day" +"%Y%m%d")

ssh -A mass-cli "moo get moose:/misc/frasia/Y$(date -d ${date_info} +"%Y").tar/*${day}* $radardir
moo get moose:/misc/frasia/Y$(date -d ${date_info}" +1 day" +"%Y").tar/*${day1}02* $radardir "

echo $radardir
cd $radardir
for file in *${day}*.tar *${day1}*.tar; do
  echo $file
  tar xf $file '*_rainrate_composite_1km_merged.Z'
  gzip -d *.Z
done
cd

# Note: this script combines pp files AND changes units from mm/hr -> kg/m2/s
ssh -A sci6 "module load jaspy; python ${user_dir}/merge_radar.py $day $radardir "

rm $radardir/temp
rm $radardir/*merged
rm $radardir/MetOffice*

echo "Done"
