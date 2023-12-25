#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# choices: mujoco131, mujoco150, mujoco200
dockerfile=$1
shift

if hash nvidia-docker 2>/dev/null; then
    nvidia-docker run -ti --rm \
            --net host \
            -v ${SCRIPT_DIR}:${SCRIPT_DIR} \
        ${USER}/${dockerfile} \
        $@
else
    docker run -ti --rm \
        --net host \
        -v ${SCRIPT_DIR}:${SCRIPT_DIR} \
        ${USER}/${dockerfile} \
        $@
fi