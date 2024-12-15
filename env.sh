#!/bin/bash
# execute script using
#     source <(sudo cat /path/to/env.sh)

conda deactivate
conda deactivate
conda env remove --yes --quiet --name digits
conda create --yes --quiet --name digits python=3.10.* ipykernel ipywidgets > /dev/null
conda activate digits
pip install numpy
pip install matplotlib
