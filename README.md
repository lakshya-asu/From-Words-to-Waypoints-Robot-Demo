# Graph_EQA
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
UPDATE

## Files requiring migration
* segmentation_colormap.py
* stretch_ai_utils package
* detection package with detic_segmenter.py and grounded_sam2.py

## TODOs
* Instructions needed for configuring paths for configs, questions, detection tools (see detic_segmenter.py)
* Instructions needed for installing Detic
* install_requires in setup.py needed

## GraphEQA Workspace Configuration
Below are instructions for how to set up a workspace to run and contribute to GraphEQA on Ubuntu 20.04.

Owners and collaborators of this repo are not claiming to have developed anything original to Hydra or any other MIT Spark lab tools.

### Setting up Hydra on Ubuntu 20.04
This set of instructions is only for local Ubuntu 20.04 installations. We unfortunately do not yet formally support other Ubuntu versions or Docker.

0) Install Hydra following the instructions at the [MIT-SPARK Lab Hydra repo](https://github.com/MIT-SPARK/Hydra?tab=readme-ov-file).

1) If you do not have conda, install it. Then create a conda environment:

``` bash
conda create -n "grapheqa" python=3.10`
```

Activate the workspace:

``` bash
conda activate grapheqa
```

2) Follow the instructions for [installing the Hydra Python bindings](https://github.com/MIT-SPARK/Hydra/blob/main/python/README.md) inside of the conda environment created above. 

3) [Install Habitat Simulator](https://github.com/facebookresearch/habitat-sim#installation).

### Download the HM3D dataset
The HM3D dataset along with semantic annotations can be downloaded [here](https://github.com/matterport/habitat-matterport-3dresearch), for example, `hm3d-train-habitat-v0.2.tar` and `hm3d-train-semantic-annots-v0.2.tar`. Update the `scene_data_path` and `semantic_annot_data_path` fields in `grapheqa.yaml` to correspond to the directories in which the above data was downloaded. See `grapheqa.yaml` as a guide.

### Clone Explore-EQA dataset
UPDATE
* Where should user put explore-eqa_semnav? This should also be changed to explore-eqa_grapheqa.

```bash
git clone git@github.com:SaumyaSaxena/explore-eqa_semnav.git -b semnav
```

Update question_data_path, init_pose_data_path, and eqa_dataset_enrich_labels to correspond to the directory in which explore-eqa_semnav was cloned.

### Install Torch
Install pytorch based on your CUDA version: https://pytorch.org/get-started/locally/

### Install Detic
Install Detic: https://github.com/facebookresearch/Detic/blob/main/docs/INSTALL.md

### Installing GraphEQA
You can now pip install GraphEQA.

```bash
git clone git@github.com:SaumyaSaxena/Graph_EQA.git
cd Graph_EQA
pip install -e .
```

The OpenAI API requires an API key. Add the following line to your .bashrc:

`export OPENAI_API_KEY=<YOUR_OPENAI_KEY>`

Google's Gemini will also need an API key, call it GOOGLE_API_KEY:

`export GOOGLE_API_KEY=<YOUR_GOOGLE_KEY>`

## Running GraphEQA

### Simulation

### On Hello Robot's Stretch