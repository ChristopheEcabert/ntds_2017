#!/bin/bash
echo "Enabling Python 3.6 ..."
source activate py36
echo "Starting jupyter notebook server..."
#jupyter nbextension enable mayavi --user --py
jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000