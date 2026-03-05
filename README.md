# Running Spatial Experiment on the ROBOTIS AI Worker

This guide details exactly how to run the multi-agent AI brain (`spatial_experiment`) directly on the physical ROBOTIS AI Worker platform, utilizing its onboard ZED 2i camera and Nav2 mobility stack.

---

## 1. Prerequisites (Host Machine)

Ensure you have cloned this repository onto the robot's main compute unit (or a tethered laptop with a strong GPU). You will need CUDA/NVIDIA drivers installed on the host.

```bash
gh repo clone lakshya-asu/From-Words-to-Waypoints-Robot-Demo
cd From-Words-to-Waypoints-Robot-Demo
```

Download any necessary LLM/VLM weights if you are running models locally. If using an API provider (Gemini, Claude, OpenAI), ensure your environment variables are set in your bash profile:

```bash
export CLAUDE_API_KEY="your_key"
```

---

## 2. Docker Environment Setup

To ensure dependency isolation between the ROS 2 physical drivers and the heavily Python-dependent AI brain, we use a unified Docker container.

### Pre-built Image (Quickstart)
If you don't want to build from scratch, you can pull the ready-to-used image directly from Docker Hub:
```bash
docker pull flux04/grapheqa_for_robotis:0.0.1
```

### Build the Image (From Scratch)
If you modify the Python dependencies or source code, navigate to the `docker` directory and run the build script:
```bash
cd mapg_real_demo/docker
./docker_build_robotis.sh
```
This will build `grapheqa_for_robotis` which contains ROS 2 Humble, the ZED SDK, PyTorch, and all required spatial reasoning libraries.

### Run the Container
We have a helper script that mounts your GPUs, passes through the X11 socket (for visualizing maps and camera feeds), and enters the container with host-networking so it can see the ROS 2 topics:

```bash
./docker_run_robotis.sh
```

*(You are now executing commands inside the Docker container)*

---

## 3. Hardware Bringup (ROS 2)

Before starting the AI brain, the physical hardware layers and sensor drivers must be active. 

*Note: You may run this in a separate terminal OR in tmux inside the same container. As long as ROS 2 can discover the topics, it works.*

**Start the low-level base node, LIDAR, and ZED 2i camera:**
```bash
ros2 launch ffw_bringup bringup.launch.py
ros2 launch ffw_bringup camera_zed.launch.py
```

**Start the Navigation 2 stack (Nav2):**
```bash
ros2 launch ffw_navigation navigation.launch.py
```

Verify that topics such as `/cmd_vel`, `/odom`, `/zed/zed_node/rgb/image_rect_color`, and `/zed/zed_node/depth/depth_registered` are actively publishing data using `ros2 topic list`.

---

## 4. Execution Mode A: Interactive Exploration

In this mode, the robot starts with no prior knowledge of the room. It will dynamically explore, build a 3D semantic `SparseVoxelMap`, actively look for objects, and answer your question.

Execute the agent script:
```bash
cd /opt/mapg_real_demo
python spatial_experiment/scripts/run_ai_worker_agent.py \
    --question "Explore the room and find the apple on the table." \
    --vlm_provider "qwen"
```

**What happens:**
1. The script initializes the AI brain and connects to the ROS 2 topics.
2. `RobotisHydraAgent` begins constructing a live 3D semantic voxel map based on the ZED 2i data stream.
3. The LLM orchestrator parses your question and directs the internal spatial planners to explore unmapped territory (`goto_frontier`).
4. The Nav2 stack safely drives the robot to these frontiers while maintaining obstacle avoidance.
5. Once found, the visual verifier agent confirms the target.

---

## 5. Execution Mode B: Offline Ablation (Pre-recorded Map)

If you wish to test reasoning capabilities without the volatility of physical mapping—or if you simply want to replay a specific scenario—you can provide a pre-recorded Dynamic Scene Graph (`dsg.json`). The robot will use this graph for planning and command base movements, but its real-time voxel updates will be disabled to keep the graph pristine.

Execute the offline ablation script:
```bash
cd /opt/mapg_real_demo
python spatial_experiment/scripts/run_ai_worker_agent_offline.py \
    --question "Find the red coffee mug." \
    --vlm_provider "gemini" \
    --scene_graph_path "spatial_experiment/zed2i/backend/dsg.json"
```

**What happens:**
1. The script loads the predefined 3D map graph from the JSON file.
2. The AI reasons over *this* graph instead of live data to formulate a plan.
3. It sends movement commands (`NavigateToPose`) to the physical base to carry out the necessary trajectory to reach the nodes specified in the offline graph.
