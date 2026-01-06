#!/bin/bash
set -e
xhost +local:docker

docker run -it --rm --gpus all \
  --net=host \
  -v "$HOME/clutter_quantification:/clutter_quantification:rw" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY="$DISPLAY" \
  --workdir /clutter_quantification \
  clutter_quantification \
  bash -lc '
    source /opt/conda/etc/profile.d/conda.sh && conda activate clutter_quantification
    export PYTHONPATH="/clutter_quantification/src:$PYTHONPATH";
    cd data_collection && python convonet_setup.py build_ext --inplace;
    exec bash
  '