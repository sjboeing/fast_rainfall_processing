import os
import datetime
import sys
import glob
import warnings
import argparse
#warnings.filterwarnings("ignore")

##################################################################
# Extra script to produce FOREWARNS 4 day flood forecast pdf. Warning: will hang without error message if any files are missing, due to quiet command on l87! Requires pdflatex.
#
# Argument parser options:
# -f : %Y%m%d string of forecast initialisation date. Only compulsory argument
# -i : %H string of forecast intialisation hour (UTC for MOGREPS-UK)
# -u : User directory (see documentation of requisite folder structure - where pdf is outputed to).
#
# PREREQ's: integrated_rainfall_processing.py ; FloodForecaseLookup.py
##################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fcst_str", required=True, type=str)
parser.add_argument("-i", "--init", default=18, required=False)
parser.add_argument("-u", "--user", default="/home/users/bmaybee/", required=False, type=str)
args = parser.parse_args()
user_root=args.user

if user_root[-1] != "/":
    user_root=user_root+"/"
fcst_str=args.fcst_str + "_" + (args.init).zfill(2)
figdir = user_root+"output_plots/forecasts/"+fcst_str[:6]+"/"+fcst_str+"/"
file_str = figdir+fcst_str+'_FOREWARNS_fcst'

fcst_date = datetime.datetime.strptime(fcst_str, "%Y%m%d_%H")
#fcst_date = fcst_date + datetime.timedelta(hours=4)
dates = []
doc_str=""
for i in range(1,5):
    date= fcst_date + datetime.timedelta(days=i)
    date_str=date.strftime("%Y%m%d")
    date=date.strftime("%d/%m/%Y")
    doc_str=doc_str+"""
\section{"""+date+r""" forecast}
\begin{frame}
\frametitle{"""+date+r""": FOREWARNS forecast}
\textbf{LH panel: Basemap; RH panel: Forecast}
\centering\includegraphics[width=0.46\textwidth, trim=0 0 0 0, clip]{"""+user_root+r"""MO_testbed_2023/basemap.png}
\centering\includegraphics[width=0.53\textwidth, trim=0 0 0 0, clip]{"""+figdir+date_str+r"""_fcst_98_EngWls_Floodplots.jpg}
\end{frame}
\begin{frame}
\frametitle{"""+date+r""": RWCRSs and FOREWARNS forecast}
\centering\includegraphics[width=0.46\textwidth, trim=25 0 15 0, clip]{"""+figdir+date_str+r"""_98_rwcrs.pdf}
\centering\includegraphics[width=0.53\textwidth, trim=0 0 0 0, clip]{"""+figdir+date_str+r"""_fcst_98_EngWls_Floodplots.jpg}
\end{frame}
\begin{frame}
\frametitle{"""+date+r""": max. accum. T=180, 98$^{th}$ pp}
\centering\includegraphics[width=0.78\textwidth]{"""+figdir+date_str+r"""_98_max_T180_accum.pdf}
\end{frame}
\begin{frame}
\frametitle{"""+date+r""": max. accum., T=180, exact}
\centering\includegraphics[width=0.78\textwidth]{"""+figdir+date_str+r"""_exact_max_T180_accum.pdf}
\end{frame}
%
    """

f = open(file_str+'.tex','w')
f.write(r"""\documentclass{beamer}
\usetheme{Boadilla}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{fontenc}
%\usepackage[ddmmyyyy]{datetime}
\title[FORWARNS forecasts]{FOREWARNS forecasts from MOGREPS-UK\\Cycle: """+str(fcst_date.hour)+r"""UTC """+fcst_date.strftime("%d/%m/%Y")+r""".}
\author[FOREWARNS]{University of Leeds\\Met Office Summer 2023 Testbed}
\date{}
\begin{document}
\frame{\titlepage}
\begin{frame}
\frametitle{Table of Contents}
\tableofcontents
\end{frame}
%""" + 
doc_str 
+ r"""
\end{document}
""")
f.close()

#os.system('ssh -A sci2-test pdflatex --output-directory='+figdir+' '+file_str+'.tex')
os.system('pdflatex -quiet --output-directory='+figdir+' '+file_str+'.tex > /dev/null 2>&1')
os.system('pdflatex -quiet --output-directory='+figdir+' '+file_str+'.tex > /dev/null 2>&1')
os.system('rm -f '+file_str+'.aux') 
os.system('rm -f '+file_str+'.log')
os.system('rm -f '+file_str+'.nav') 
os.system('rm -f '+file_str+'.out')
os.system('rm -f '+file_str+'.snm') 
os.system('rm -f '+file_str+'.toc')
os.system('rm -f '+file_str+'.tex')