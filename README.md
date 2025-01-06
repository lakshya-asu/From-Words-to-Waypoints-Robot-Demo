# Graph_EQA
This repo provides code for GraphEQA, a novel approach for utilizing 3D scene graphs for embodied question answering (EQA), introduced in the paper [GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering](https://www.arxiv.org/abs/2412.14480).

<div align="center">
    <img src="doc/grapheqa.gif">
</div>

Please cite using the following if you find GraphEQA relevant or useful for your research.

```bibtex
@misc{saxena2024grapheqausing3dsemantic,
      title={GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering}, 
      author={Saumya Saxena and Blake Buchanan and Chris Paxton and Bingqing Chen and Narunas Vaskevicius and Luigi Palmieri and Jonathan Francis and Oliver Kroemer},
      year={2024},
      eprint={2412.14480},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2412.14480}, 
}
```

#### Funding and Disclaimer
UPDATE

## Files requiring migration
* segmentation_colormap.py
* stretch_ai_utils package
* detection package with detic_segmenter.py and grounded_sam2.py

## TODOs
* Instructions needed for configuring paths for configs, questions, detection tools (see detic_segmenter.py)
* Instructions needed for installing Detic

## GraphEQA Workspace Configuration
Below are instructions for how to set up a workspace to run and contribute to GraphEQA on Ubuntu 20.04.

Owners and collaborators of this repo are not claiming to have developed anything original to Hydra or any other MIT Spark lab tools.

### Setting up Hydra on Ubuntu 20.04
This set of instructions is only for local Ubuntu 20.04 installations. We unfortunately do not yet formally support other Ubuntu versions or Docker.

0) If you don't have ROS Noetic, install it: https://wiki.ros.org/ROS/Installation

1) Then do:

``` bash
sudo apt install python3-rosdep python3-catkin-tools python3-vcstool
```

Set up rosdep:

``` bash
sudo rosdep init
rosdep update
```

Set up a catkin workspace for building MIT Spark Lab's Hydra:

``` bash
mkdir -p catkin_ws_grapheqa/src
```

Then `cd` into it:

``` bash
cd catkin_ws_grapheqa
```

Install our Fork of Hydra. This will also install forks of Spark-DSG and Hydra-ROS via the branches specified in our modified `hydra.rosinstall`.

``` bash
source /opt/ros/noetic/setup.bash
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release

cd src
git clone git@github.com:blakerbuchanan/Hydra.git hydra
vcs import . < hydra/install/hydra.rosinstall
rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin build
```

At this point we can make sure Hydra is installed correctly by just trying to run it.

``` bash
source devel/setup.bash
roslaunch hydra_ros uhumans2.launch
```

An RViz window should open. If nothing crashes, you are probably good.

To test further, download the uhumans2 dataset at https://drive.usercontent.google.com/download?id=1CA_1Awu-bewJKpDrILzWok_H_6cOkGDb&authuser=0 .

Then do:

``` bash
rosbag play path/to/rosbag --clock
```

You should see the scene graph, mesh, etc., begin populating in the RViz window that opened. Note that this is just the test and default launch provided by MIT Spark lab.

2) Install the Hydra Python bindings

If you do not have conda, install it. Then create a conda environment:

``` bash
conda create -n "grapheqa" python=3.9`
```

Activate the workspace:

``` bash
conda activate grapheqa
```

Now we will install editable versions of the Spark-DSG and Hydra Python bindings within the conda environment:

``` bash
# required to expose DSG python bindings
pip install -e "src/spark_dsg[viz]"
cd src/hydra
pip install -r python/build_requirements.txt
pip install -e .
```

3) Install Habitat via conda: https://github.com/facebookresearch/habitat-sim#installation

4) We need a few other things:

``` bash
pip install rerun-sdk opencv-python openai omegaconf ipdb torch torchvision transformers scikit-image yacs gpustat
pip install -q -U google-generativeai
```

The OpenAI API requires an API key. Add the following line to your .bashrc:

`export OPENAI_API_KEY=<YOUR_OPENAI_KEY>`

Google's Gemini will also need an API key, call it GOOGLE_API_KEY:

`export GOOGLE_API_KEY=<YOUR_GOOGLE_KEY>`

### Installing GraphEQA
UPDATE