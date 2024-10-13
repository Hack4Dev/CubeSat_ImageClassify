# CubeSat_ImageClassify

## Description

Welcome to the CubeSat Image Classify Hackathon project! This project includes the following Notebooks:

- *Notebook 1*: Introduction to the problem and an overview of the hackathon 
- *Notebook 2*: Reading and analyzing the astronomical data
- *Notebook 3*: Classification using a machine learning model
- *Notebook 4*: Classification using a deep learning model


## Data

The data used in this hackathon can be found at [link: coming soon]. It contains approximately 16,000 images, each with a size of 3x512x512. The images are classified into the following categories:

0. Blurry
1. Corrupt
2. Missing_Data
3. Noisy
4. Priority


## Hackathon Task

Develop a machine learning model that accurately classifies data captured by CubeSats. The goal is to prioritize which images are most valuable for transmission back to Earth, given the limited onboard resources and slow data downlink speeds. Your task is to create a lightweight model that improves the efficiency and/or classification accuracy of the existing solution in this [paper](https://arxiv.org/pdf/2408.14865).


## Prerequisites

All the necessary libraries and dependencies to run the notebooks are listed in the [requirements.txt](https://github.com/Hack4Dev/CubeSat_ImageClassify/blob/main/requirements.txt) file.

### Installation

#### On Your Local Machine

To install a single package, use the following command:

```bash
pip install --user <package>
```

To install all required packages, run:
```bash
pip install -r requirements.txt
```

On the Ilifu Cloud Computing System:

If you are participating through our cloud computing system [ilifu](https://www.ilifu.ac.za/), you can install Python libraries locally using the following command:
```bash
/shared/venv/bin/python -m pip install --user <package>
```

### Would you like to clone this repository? Feel free!

```bash
git clone https://github.com/Hack4Dev/CubeSat_ImageClassify.git
```

Then make sure you have the right Python libraries for the notebooks. 

### New to Github?

The easiest way to get all of the lecture and tutorial material is to clone this repository. To do this you need git installed on your laptop. If you're working on Linux you can install git using apt-get (you might need to use sudo):

```
apt install git
```

You can then clone the repository by typing:

```
git clone https://github.com/Hack4Dev/CubeSat_ImageClassify.git
```

To update your clone if changes are made, use:

```
cd CubeSat_ImageClassify/
git pull
```

### Original research work:

Chatar, Keenan AA, et al. "Data downlink prioritization using image classification on-board a 6U CubeSat." Sensors, Systems, and Next-Generation Satellites XXVII. Vol. 12729. SPIE, 2023. [link:](https://arxiv.org/pdf/2408.1486).


### Data used (coming soon)

