#!/bin/bash
# Script to run the physical robot Docker image

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/docker_tag.sh"

workspace_dir="$(cd "${DIR}/.." && pwd)"
docker_registry_image=flux04/grapheqa_for_robotis

echo "Running Docker Image: ${docker_registry_image}:${TAG}"

# Ensure X server accepts local connections
xhost +local:root

docker run -it --rm \
    --name grapheqa_robotis_container \
    --gpus all \
    --network host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CLAUDE_API_KEY=$CLAUDE_API_KEY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${workspace_dir}:/opt/mapg_real_demo \
    -v /dev:/dev \
    ${docker_registry_image}:${TAG} \
    /bin/bash
