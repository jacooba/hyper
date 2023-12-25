#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# choices: mujoco131, mujoco150, mujoco200
dockerfile=$1
shift

gpu=$1
shift
docker run -ti --rm \
        --env CUDA_VISIBLE_DEVICES=${gpu} \
        --gpus all \
        --net host \
        -v ${SCRIPT_DIR}:${SCRIPT_DIR} \
        ${USER}/${dockerfile} \
        $@
