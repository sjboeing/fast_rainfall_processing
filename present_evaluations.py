import os
import datetime
import sys
import glob
import warnings
import argparse
#warnings.filterwarnings("ignore")

##################################################################
# Extra script to produce FOREWARNS forecast evaluation pdf. Requires both radar observations/flood proxies and forecast from day before initialised at time -i.
#Warning: will hang without error message if any files are missing, due to quiet command on l93! Requires pdflatex.
#
# Argument parser options:
# -f : %Y%m%d string of forecast evaluation date (i.e. validity date, NOT initialisation). Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK); to go beyond lead time of 1 day need to add 24 hour increments (see l34)
# -u : User directory (see documentation of requisite folder structure - where pdf is outputed to).
#
# PREREQ's: forecast_plots.py -e True ; FloodForecastLookup.py -r True
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--date", required=True, type=str)
parser.add_argument("-i", "--init", required=False, default=18, type=int)
parser.add_argument("-u", "--user", default="/home/users/bmaybee/", required=False, type=str)
args = parser.parse_args()
user_root=args.user
if user_root[-1] != "/":
    user_root=user_root+"/"
date_str=args.date
#eval_dir is assumed to have been created already via forecast_plots.py in evaluation mode (-e); need to run that script to generate images required by beamer template below.
eval_dir = user_root+"output_plots/evaluation/"+date_str+"/"
rad_dir = user_root+"output_plots/radar/"+date_str[:6]+"/"
file_str = eval_dir+date_str+'_{}Z_FOREWARNS_evals'.format(args.init)

date = datetime.datetime.strptime(date_str, "%Y%m%d")
fcsts=[]
fcsts.append(date - datetime.timedelta(hours=(24-args.init)))
doc_str=""
for fcst in fcsts:
    fcst_str=fcst.strftime("%Y%m%d_%H")
    fcst_dir = user_root+"output_plots/forecasts/"+fcst_str[:6]+"/"+fcst_str+"/"
    fcst="%02dZ %02d/%02d/%04d" % (fcst.hour,fcst.day,fcst.month,fcst.year)
    doc_str=doc_str+"""
\section{"""+fcst+r""" forecast}
\begin{frame}
\frametitle{"""+fcst+r""": FOREWARNS evaluation}
%\centering{\large Forecast \hspace{20pt} Flood proxy}
\centering\includegraphics[width=0.49\textwidth]{"""+fcst_dir+date_str+r"""_fcst_98_EngWls_Floodplots.jpg}
\centering\includegraphics[width=0.49\textwidth]{"""+rad_dir+date_str+r"""_rad_98_EngWls_Floodplots.jpg}
\end{frame}
\begin{frame}
\frametitle{"""+fcst+r""": RWCRS evaluation (and reference basemap)}
\centering\includegraphics[width=0.7\textwidth]{"""+eval_dir+fcst_str+r"""_98_rwcrs.pdf}
\centering\includegraphics[width=0.29\textwidth, trim=0 0 0 0, clip]{"""+user_root+r"""MO_testbed_2023/basemap.png}
\end{frame}
\begin{frame}
\frametitle{"""+fcst+r""": (r30, p98, T180)}
\centering\includegraphics[width=0.75\textwidth]{"""+eval_dir+fcst_str+r"""_98_max_T180_accum.pdf}
\end{frame}
\begin{frame}
\frametitle{"""+fcst+r""": max. accum., T=180, exact}
\centering\includegraphics[width=0.78\textwidth]{"""+eval_dir+fcst_str+r"""_exact_max_T180_accum.pdf}
\end{frame}
%
    """

date = date.strftime("%d/%m/%Y")
f = open(file_str+'.tex','w')
f.write(r"""\documentclass{beamer}
\usetheme{Boadilla}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{fontenc}
%\usepackage[ddmmyyyy]{datetime}
\title[FORWARNS evaluation]{FOREWARNS evaluation: """+date+r"""}
\author[Univ. Leeds]{University of Leeds\\Met Office Summer 2023 Testbed}
\date["""+date+r"""]{}
\begin{document}
\frame{\titlepage}
\begin{frame}
\frametitle{Table of Contents}
\tableofcontents
\end{frame}
\begin{frame}
\frametitle{"""+date+r""" radar observations}
\centering\includegraphics[width=\textwidth]{"""+rad_dir+date_str+r"""_radar_accum_plots.pdf}
\end{frame}
""" + 
doc_str 
+ r"""
\end{document}
""")
f.close()

#os.system('ssh -A sci2-test pdflatex --output-directory='+eval_dir+' '+file_str+'.tex')
os.system('pdflatex -quiet --output-directory='+eval_dir+' '+file_str+'.tex > /dev/null 2>&1')
os.system('pdflatex -quiet --output-directory='+eval_dir+' '+file_str+'.tex > /dev/null 2>&1')
os.system('rm -f '+file_str+'.aux') 
os.system('rm -f '+file_str+'.log')
os.system('rm -f '+file_str+'.nav') 
os.system('rm -f '+file_str+'.out')
os.system('rm -f '+file_str+'.snm') 
os.system('rm -f '+file_str+'.toc')
os.system('rm -f '+file_str+'.tex')