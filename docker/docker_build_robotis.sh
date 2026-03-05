#!/bin/bash
# Script to build the physical robot Docker image

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/docker_tag.sh"

docker_registry_image=flux04/grapheqa_for_robotis

echo "Building Docker Image: ${docker_registry_image}:${TAG}"

docker build \
       --platform linux/amd64 \
       -t ${docker_registry_image}:${TAG} \
       -f Dockerfile.robotis .
