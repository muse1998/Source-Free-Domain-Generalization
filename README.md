# Source-Free-Domain-Generalization
An open-world scenario domain generalization code base

## 
You can download the .zip file for all code directly from [here](https://anonfiles.com/PaR55aCay5/Source-Free-Domain-Generalization-main_zip).
Anonymous links are for the convenience of double-blind review and will not be LTS.


## Installation

```bash
# create virtual environment and install packages
conda env create -f environment.yaml

# activate virtual environment
conda activate cae

# install tllib 
python setup.py install
```



All the dataset used can be download from [google drive](???)

We offered anonymous links [PACS](https://anonfiles.com/cb54peBby2/PACS_zip), [Terra](https://anonfiles.com/s563ccC9y3/Terra_zip) for review.

you should put *.zip files in ./data and unzip.



```
└─data
    ├─domainnet
    ├─office-home
    ├─PACS
    ├─Terra
    └─VLCS
```


## Usage

### Scripts for  experiment.

```bash
# exp on all the datasets 
sh [DG_method].sh

# for example
sh cae.sh
```

### SFDG experiment.

```bash
# SFDG experiments 
python cae.py [data_path] -d [dataset] -t [target domain] -a [backbone_of_CLIP] --seed [seed] --log [log_path]

# for example 
python cae.py data/PACS -d PACS  -t S -a vitb16 --seed 0 --log logs/cae/PACS_S
```
### Open-world experiment
We collected two extra domains ('**X**' for pixel_style and '**G**' for geometric) for PACS dataset to test open-world performance of our method.

```bash
# SFDG
python cae.py data/PACS -d PACS  -t X -a vitb16 --seed 0 --log logs/cae/PACS_X

# DG
python erm.py data/PACS -d PACS  -s P A C -t G -a resnet50 --seed 0 --log logs/erm/PACS_G
```


### DG experiment.

```bash
# DG experiments  
python [DG_method].py [data_path] -d [dataset] -s [source domains] -t [target domain] -a [backbone_of_CLIP] --seed [seed] --log [log_path]

# for example 
python erm.py data/PACS -d PACS -s P A C  -t S -a resnet50 --seed 0 --log logs/cae/PACS_S --freeze-bn
```



### TLlib experiment.
*TLlib* is a public toolbox for transfer learning, we modified these files for experiments on Terra and VLCS datasets.

./tllib/vision/datasets/terra.py 

./tllib/vision/datasets/vlcs.py

./tllib/vision/datasets/\_\_init\_\_.py 

