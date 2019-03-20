# Image Based Tracker

This code can be used to track scenes by providing 2D RGB images. It uses the ORB feature detector which detects and computes keypoints and their descriptors in the input images. These descriptors of two consecutive are then matched and a transformation model is calculated.
The code in this repository is specialised to track patients movement during surgery. In order to provide robust and accurate tracking we assume that the scene to track is a rigid body.
The code can deal with surgical instruments in the field of view. We use semantic segmentation to exclude these instruments from the keypoint search.

This repository also provides a class to plot the tracking results. See the documentation for more information.

## Usage
This repository provides two example files `ImageDemo.py` and `VideoDemo.py` to demonstrate the usage of the tracking and plotting code.\
See the documentation if you want to use the framework in your own program.

### Dependencies
The code is written using Python3.\
I recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required dependencies. Using pip (or pip3) can lead to trouble when you try to run the tensorflow packages. Depending on your machine some additional drivers are needed (which gave me a headache trying to figure out why tensorflow was not working) and conda handles that for you.\
You can use the file `data/thesis.yml` to create a conda environment with all dependencies already installed.\
Using conda you can simply run `conda env create -f [path/to/thesis.yml]`.\
If you want to use pip anyway see `data/dependencies.txt` for all the required packages.\
If your machine has a physical GPU install `tensorflow-gpu` as well to run the segmentation on your GPU (this will speed it up a lot).


## Code Documentation

To generate the documentation files run `doxygen Doxyfile` in the projects root directory. A new directory `docs` will be created containing the file `annotated.html`.

## Semantic Segmentation
All code in the `src/Segmentation` directory is based on [George Seif's Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite).\
In order to use semantic segmentation during tracking you need a pretrained neural net. Since Git does not allow files of size larger than 100MB this pretrained model cannot be provided here. If you are interested in using the model for surgical instrument segmentation text me. I trained it using the above mentioned GitHub repository and image data from the EndoVis'15 Instrument Subchallenge  [Dataset](http://opencas.webarchiv.kit.edu/?q=node/30).  

### Change the tracking task or the model
If you want to feed the tracking code you own neural net (e.g. for scenes other than surgical) it is best to train it using the above mentioned GitHub repository. A detailed description on how to train it can be found in its Readme.\
I use Googles neural net model `DeepLab v3+` for training and prediction. If you want to use a different model (see the Segmentation [Repo](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) what models are available) you have to change some code in `src/Tracker.py`. Change the arguments in the `__init__` method according to your used model and trained net.\
Place the trained model in a directory `src/Segmentation/trained_net`. Make sure you place the `class_dict.csv` file from training in `src/Segmentation/utils`.

Since I only tested the code using the `DeepLab v3+` model, I can't guarantee the compatibility of other models, you might have to change some other code. If you have trouble doing so dm me and I will see if I can help you.
