# GraphEQA
This repo provides code for GraphEQA, a novel approach for utilizing 3D scene graphs for embodied question answering (EQA), introduced in the paper [GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering](https://www.arxiv.org/abs/2412.14480).

<div align="center">
    <img src="doc/grapheqa.gif">
</div>

* Website: https://saumyasaxena.github.io/grapheqa/
* arXiv: https://www.arxiv.org/abs/2412.14480
* Paper: https://saumyasaxena.github.io/grapheqa/grapheqa_2025.pdf

If you find GraphEQA relevant or useful for your research, please use the following citation:

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
This work was in part supported by the National Science Foundation
under Grant No. CMMI-1925130 and in part by the EU Horizon 2020
research and innovation program under grant agreement No. 101017274
(DARKO). Any opinions, findings, and conclusions or recommendations
expressed in this material are those of the author(s) and do not necessarily
reflect the views of the NSF.

## GraphEQA Workspace Configuration
Below are instructions for how to set up a workspace to run and contribute to GraphEQA on Ubuntu 20.04.

Owners and collaborators of this repo are not claiming to have developed anything original to Hydra or any other MIT Spark lab tools.

### Setting up Hydra on Ubuntu 20.04
This set of instructions is only for local Ubuntu 20.04 installations.

0) Install our fork of Hydra following the instructions at [this link](https://github.com/blakerbuchanan/Hydra). You will need to clone the `grapheqa` branch of this fork.

1) If you do not have conda, install it. Then create a conda environment:

``` bash
conda create -n "grapheqa" python=3.10`
```

Activate the workspace:

``` bash
conda activate grapheqa
```

2) Follow the instructions for [installing the Hydra Python bindings](https://github.com/MIT-SPARK/Hydra/blob/main/python/README.md) inside of the conda environment created above. Before installing, be sure to source `devel/setup.bash` in the above catkin workspace, otherwise the installation of the python bindings will fail.

3) [Install Habitat Simulator](https://github.com/facebookresearch/habitat-sim#installation).

### Download the HM3D dataset
The HM3D dataset along with semantic annotations can be downloaded [here](https://github.com/matterport/habitat-matterport-3dresearch), for example, `hm3d-train-habitat-v0.2.tar` and `hm3d-train-semantic-annots-v0.2.tar`. Update the `scene_data_path` and `semantic_annot_data_path` fields in `grapheqa.yaml` to correspond to the directories in which the above data was downloaded. See `grapheqa.yaml` as a guide.

### Get relevant Explore EQA files
Navigate to [this repo](https://github.com/SaumyaSaxena/explore-eqa_semnav/tree/master/data) and download `questions.csv` and `scene_init_poses.csv` into a directory in your workspace.  

Update `question_data_path` and `init_pose_data_path` in the `grapheqa_habitat.yaml` to correspond to the directory in which you downloaded the above two files.

### Install the Stretch AI package from Hello Robot
Follow the install instructions at our fork of `stretch_ai` [found here](https://github.com/blakerbuchanan/stretch_ai) to install the packages necessary to run GraphEQA on Hello Robot's Stretch platform.

### Installing GraphEQA
Clone and install GraphEQA:

```bash
git clone git@github.com:SaumyaSaxena/Graph_EQA.git
cd Graph_EQA
pip install -e .
```

The OpenAI API requires an API key. Add the following line to your .bashrc:

`export OPENAI_API_KEY=<YOUR_OPENAI_KEY>`

Google's Gemini will also need an API key, call it GOOGLE_API_KEY:

`export GOOGLE_API_KEY=<YOUR_GOOGLE_KEY>`

## Running GraphEQA with habitat
To run GraphEQA with Habitat Sim, run:
```bash
python scripts/run_vlm_planner_eqa_habitat.py -cf grapheqa_habitat
```
This will run GraphEQA on the hm3d dataset, with results available in the `outputs` directory.

## On Hello Robot's Stretch
To run GraphEQA on Hello Robot's Stretch platform, you will need to run the server on the Stretch robot following the instructions at [this fork](https://github.com/blakerbuchanan/stretch_ai). Once you have successfully launched the server, open a terminal on your computer (client side) and run:

```bash
python scripts/run_vlm_planner_eqa_stretch.py -cf grapheqa_stretch
```

This will run GraphEQA on Hello Robot's Stretch.