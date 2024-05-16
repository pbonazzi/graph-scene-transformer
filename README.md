#  3D Scene Generation from Scene Graphs and Self-A!entions 

### [💻 Blog](https://pietrobonazzi.com/projects/scenegraph) |[📜 Paper](https://arxiv.org/pdf/2404.01887v2) 


We present a conditional VAE architecture with self-attention layers as fundamental building blocks, tailored to 3D scene generation from scene graphs.

<p align="center">
  <img width="800" src="https://github.com/uzh-rpg/scene_graph_3d/blob/main/docs/assets/base.png?style=centerme">
</p>

## Navigation on the repo

The project structured as following :

    .
    ├── config                              # configuration files
    ├── input                               # inputs for the model
    ├── docs                                # assets, instructions, requirements file for virtual environments
    ├── network                             # model, layers, metrics
    ├── outputs                             # results (checkpoints, saved configigurations)
    ├── scripts                             # scripts for inference, training and evalution
    ├── main.py                             # parsers and data loader
    ├── README.md                           
    .

## Setup

[Follow these instructions](./docs/installation.md) to set up the virtual environment.

## Data
Download the <a href="https://shapenet.org/download/shapenetsem" >ShapeNetSem</a> dataset to the `input/data/shapenet_sem` folder

## Training
To train our main model (GPH) run:
```
CONFIG_PATH=config/1_GPH.json
SHAPENET_PATH="input/data/shapenet_sem"
python3 main.py --c $CONFIG_PATH --sp $SHAPENET_PATH
```

## Evaluation
To inference on scene graphs and floor plans, change the `inference` parameter in config file to be `True` and run same codes above.

## Reference our work

```
@article{uzh2021transp,
  title={3D Scene Generation with Scene Graphs Leveraging Self-Attention},
  author={Bonazzi, Wang, Arroyo, Manhard, Messikomer, Scaramuzza, Tombari},
  year={2022}
}
```
