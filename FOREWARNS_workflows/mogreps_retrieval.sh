#!/bin/bash
# PREREQ's : access to MASS mogreps-uk files (requires Met Office approval)

fcst_date=$(date -d "$1" +"%F")
init=$2
echo $fcst_date, $init
#If script run in isolation need to define gws_dir variable; inherited from master.sh.
#gws_dir=/gws/nopw/j04/icasp_swf/bmaybee
mkdir ${gws_dir}/mogreps_ensembles/$(date -d ${fcst_date} +"%Y%m")
#Pull from MASS to a temporary directory; when all files in place and processed move to archive location.
datadir=${gws_dir}/mogreps_ensembles/$(date -d ${fcst_date} +"%Y%m")/temp
mkdir $datadir
rm $datadir/*
    
mass-pull () {
for j in $(seq $(($init-5)) $init ); do
  i=$(date --date="${fcst_date} +$j hours" +"%Y%m%d_%H")
  echo $i
  touch $datadir/query
  cat >$datadir/query <<EOF
  begin
    filename = "prods_op_mogreps-uk_${i}_*"
    stash = 04203
  end
EOF
#Note: 04203 is UM variable "stratiform rainfall rate (kg m^3 / s)". Covers all rainfall for CP ensemble.

  ssh -A mass-cli.jasmin.ac.uk moo select $datadir/query moose:/opfc/atm/mogreps-uk/prods/$(date -d ${fcst_date} +"%Y%m").pp $datadir
done 

rm $datadir/MetOffice*
rm $datadir/query
}

#Pull 18 member mogreps ensemble from MASS archive
mass-pull

#Commented python calls generate single .nc file for each ensemble member from multiple pp files. Very slow for England + Wales - leave as .pp filelist.
if [[ $(ls $datadir/*003.pp | wc -l) == 18 ]]
then 
  echo "Mass pull successful"
  #ssh -A sci3 "module load jaspy; python ${user_dir}/get_mogreps_ensemble.py $datadir"
elif [[ $(ls $datadir/*003.pp | wc -l) < 18 ]]
then
  echo "First mass-pull did not retrieve enough files; retrying"
  sleep 10m
  mass-pull
  #ssh -A sci6 "module load jaspy; python ${user_dir}/get_mogreps_ensemble.py $datadir"
elif [[ $(ls $datadir/*003.pp | wc -l) > 18 ]]
then
  echo "Something was wrong on mass, too many files available. No processing invoked."
fi