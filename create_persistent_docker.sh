docker run --gpus all -it --name genesis-dev   -e DISPLAY=$DISPLAY   -v /dev/dri:/dev/dri   -v /tmp/.X11-unix/:/tmp/.X11-unix   -v "$PWD":/workspace   genesis:latest
