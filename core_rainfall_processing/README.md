# fast_rainfall_processing

A numba implementation of fast neighboorhood percentile processing

Sped up by keeping track of already sorted values

Optimised for memory usage

Requires numba and iris

Requires access to relevant MOGREPS data

Postprocessing of radar data still missing

Usage
```
python standalone_rainfall_processing_subdomain.py
```

Optional arguments
```
  -h, --help            show this help message and exit
  -w WINDOW_LENGTH, --window_length WINDOW_LENGTH
```
