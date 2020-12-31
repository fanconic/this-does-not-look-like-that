# **This does not look like That**: Evaluating theInterpretability of Deep Models

Group Project for the Deep Learning Lecture in Fall 2020 at ETH Zürich by
- Adrian Hoffmann - adriahof@student.ethz.ch
- Claudio Fanconi - fanconic@student.ethz.ch
- Rahul Rade - rarade@student.ethz.ch

## Reproduce Results of Experiments:
To recreate the results of the experiments you can use the provided Python Notebooks.

Download the code from this repository:
```git clone https://gitlab.ethz.ch/fanconic/this-does-not-look-like-that```
```cd this-does-not-look-like-that```

We suggest to create a virtual environment for this:
```conda create --name experiment_env```
```conda activate experiment_env```

Install the Requirements
```pip install -r requirements.txt```

Download the dataset and model weights from here:
CUB 200 Dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
Generally Trained ProtoPNet Weights:
JPEG Experiment ProtoPNet Weights: https://drive.google.com/file/d/1AbFmDI3E3eSc8zU5h-71HOitPMWIi3zV/view?usp=sharing

### File Descriptons

##### Notebooks

- `run_training.ipynb`: Train ProtopNet on Google Colab.
- `local_analysis_attack2.ipynb`: Attack 2 (Make head disappear).
- `local_analysis_attack3.ipynb`: Attack 3 (Make stomach similar as head).

##### Python
- `img_aug.py`: Augemtation.
- `main.py`: Training.
- `settings.py`: Settings, paths and configuration.


#

CUB-200-2011 can be downloaded from:
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
