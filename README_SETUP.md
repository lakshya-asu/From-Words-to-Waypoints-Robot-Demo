# GraphEQA Habitat Setup (Docker + Datasets + Test Run)

This guide helps you set up and run GraphEQA Habitat in a Docker container with GPU support.

## Links (reference)

```text
Datasets (Google Drive folder):
https://drive.google.com/drive/folders/1YiWecgga3Eh7GWsdlQEv2iaklzO6MaQ7

Repo:
https://github.com/TechTinkerPradhan/graph_eqa_swagat

Docker image:
https://hub.docker.com/r/swagatpradhan/grapheqa_habitat_swagat
```

## What you will end up with

- Repo on host: `~/graph_eqa_swagat/`
- Datasets on host: `~/graph_eqa_swagat/datasets/`
- Docker container name: `grapheqa_dev`
- Inside container:
  - Repo mounted at `/root/graph_eqa`
  - Datasets mounted at `/datasets`
  - Conda env: `grapheqa`
  - Test run command works

---

## 0) Prerequisites

This assumes:
- Ubuntu 22.04 or similar Linux
- An NVIDIA GPU + working host driver
- Docker Engine + Docker Compose plugin installed
- GPU support in Docker (NVIDIA Container Toolkit)

### Quick checks (host)

```bash
nvidia-smi
docker --version
docker compose version
```

Optional but recommended GPU check for Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

If the above fails, fix GPU-in-Docker before proceeding.

---

## 1) Clone the repo

If you already have Git installed, skip the install line.

```bash
sudo apt update
sudo apt install -y git
```

Clone:

```bash
cd ~
git clone https://github.com/TechTinkerPradhan/graph_eqa_swagat
cd graph_eqa_swagat
```

If the repo uses submodules, also run:

```bash
git submodule update --init --recursive
```

---

## 2) Download and extract datasets

Download the Google Drive folder and place it under this repo as:

```text
~/graph_eqa_swagat/datasets/
```

There are multiple ways to do this. Use whichever is easiest for you.

### Option A: Download via browser
1. Open the Google Drive link (above).
2. Download the folder (Google may zip it for you).
3. Extract it so that `datasets/` exists in the repo root.

You want this kind of layout:

```bash
cd ~/graph_eqa_swagat
ls -lah datasets
```

You should see dataset folders inside.

### Option B: Use rclone (recommended for large Drive folders)
Install rclone:

```bash
sudo apt update
sudo apt install -y rclone
```

Configure it (interactive):

```bash
rclone config
```

Then copy the Drive folder to your repo `datasets/`:

```bash
cd ~/graph_eqa_swagat
mkdir -p datasets
# After config, you will have a remote name like "gdrive:"
# Use rclone lsd gdrive: to find the right folder path on the remote.
rclone copy gdrive:"PATH/TO/1YiWecgga3Eh7GWsdlQEv2iaklzO6MaQ7" ./datasets --progress
```

If you do not know the remote path, use:
```bash
rclone lsd gdrive:
rclone ls gdrive: --max-depth 2
```

---

## 3) Pull the Docker image

```bash
docker pull swagatpradhan/grapheqa_habitat_swagat:v1
```

Verify:
```bash
docker images | grep -i grapheqa
```

---

## 4) Create or update `docker-compose.yml`

The main tricky part is bind-mounts. Use a compose file that is portable.

From the repo root:

```bash
cd ~/graph_eqa_swagat
```

Use this `docker-compose.yml`:

```yaml
services:
  grapheqa:
    image: swagatpradhan/grapheqa_habitat_swagat:v1
    container_name: grapheqa_dev

    # Prefer this if your docker compose supports it
    gpus: all

    # If `gpus: all` fails on your machine, comment it and use:
    # runtime: nvidia

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
      - DISPLAY=${DISPLAY}
      - QT_X11_NO_MITSHM=1
      - QT_QPA_PLATFORM=xcb

      # Put secrets in a .env file, do not hardcode them here
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}

    volumes:
      # Portable mounts: relative to the repo root
      - ./:/root/graph_eqa
      - ./datasets:/datasets

      # Optional: X11 support for GUI apps
      - /tmp/.X11-unix:/tmp/.X11-unix:rw

    working_dir: /root/graph_eqa
    tty: true
    stdin_open: true
```

### Create a `.env` file for the API key

From repo root:

```bash
cd ~/graph_eqa_swagat
nano .env
```

Add:

```bash
GOOGLE_API_KEY=PASTE_YOUR_KEY_HERE
```

Make sure `.env` is ignored by git (it should be, but confirm):

```bash
cat .gitignore | grep -n ".env" || true
```

If not present, add `.env` to `.gitignore`.

---

## 5) Start the container

From repo root:

```bash
cd ~/graph_eqa_swagat
docker compose up -d
```

If you previously ran an older compose and mounts look wrong, recreate:

```bash
docker compose down -v
docker rm -f grapheqa_dev 2>/dev/null || true
docker compose up -d --force-recreate
```

---

## 6) Verify the bindings and GPU

### Check mounts from the host

```bash
docker inspect grapheqa_dev --format '{{range .Mounts}}{{println .Type .Source "->" .Destination}}{{end}}'
```

You want to see something like:

```text
bind /home/USERNAME/graph_eqa_swagat -> /root/graph_eqa
bind /home/USERNAME/graph_eqa_swagat/datasets -> /datasets
```

### Check inside the container

```bash
docker exec -it grapheqa_dev bash
```

Inside:

```bash
ls -lah /root/graph_eqa | head -n 30
ls -lah /datasets | head -n 30
nvidia-smi
```

If `/root/graph_eqa` looks empty, your bind mount is not applied. Go back to section 5 and recreate the container.

---

## 7) Activate conda env and run the test benchmark

Inside the container:

```bash
conda activate grapheqa
cd /root/graph_eqa
```

Confirm the script exists:

```bash
ls -lah spatial_experiment/scripts | head -n 50
ls -lah spatial_experiment/scripts/run_vlm_planner_eqa_habitat_benchmark.py
```

Run:

```bash
python spatial_experiment/scripts/run_vlm_planner_eqa_habitat_benchmark.py -cf grapheqa_habitat_benchmark
```

---

## 8) Useful container commands

Logs:
```bash
docker compose logs -f --tail=200
```

Stop:
```bash
docker compose down
```

Shell:
```bash
docker exec -it grapheqa_dev bash
```

---

## Troubleshooting

### Container name conflict: "/grapheqa_dev is already in use"
```bash
docker rm -f grapheqa_dev
docker compose up -d --force-recreate
```

### Bind mounts not updating after editing compose
Mounts do not change on an existing container. Recreate:
```bash
docker compose down -v
docker rm -f grapheqa_dev 2>/dev/null || true
docker compose up -d --force-recreate
```

### GUI issues (DISPLAY / X11)
On host:
```bash
echo $DISPLAY
xhost +local:docker
```

To undo:
```bash
xhost -local:docker
```

### Docker GPU not working
Host must have a working NVIDIA driver and the NVIDIA Container Toolkit installed. Confirm:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

If the second command fails, fix that first.

---

## Notes for labmates

- Keep datasets outside the container. Always mount them to `/datasets`.
- Always run compose from the repo root so relative mounts work.
- Do not commit API keys. Use `.env`.
