#!/bin/bash

# Read in name of Dockerfile (options see below)
dockerfile=$1
shift

# Create temporary Dockerfile (passing current user ID for permissions)
(sed -e "s/<<UID>>/${UID}/" < $dockerfile ) > "${dockerfile}_UID"

# The repository name of the Docker image
user="$USER"
if [ "$dockerfile" == "Dockerfile" ]; then
    NAME="$USER"/"default"
elif [ "$dockerfile" == "Dockerfile_mj131" ]; then
    NAME="$USER"/"mujoco131"
elif [ "$dockerfile" == "Dockerfile_mj150" ]; then
    NAME="$USER"/"mujoco150"
elif [ "$dockerfile" == "Dockerfile_mj200" ]; then
    NAME="$USER"/"mujoco200"
else
    echo "--- WARNING --- Dockerfile not found."
fi

# Build Docker image
docker build -t $NAME -f "${dockerfile}_UID" .

# Remove temporary Dockerfile
rm "${dockerfile}_UID"
