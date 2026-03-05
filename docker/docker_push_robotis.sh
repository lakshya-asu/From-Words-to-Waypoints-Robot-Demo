#!/bin/bash
# Script to push the physical robot Docker image to Docker Hub

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/docker_tag.sh"

docker_registry_image=flux04/grapheqa_for_robotis

echo "Pushing Docker Image: ${docker_registry_image}:${TAG}"

# Ensure we are logged in
docker login

# Push the image
docker push ${docker_registry_image}:${TAG}

echo "Successfully pushed ${docker_registry_image}:${TAG} to Docker Hub."
