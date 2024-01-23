# sunine-DA_MG: Multi-genre domain alignment development version of [sunine](https://gitlab.com/csltstu/sunine).

## Quick installation
1. Clone this repo

```base
git clone https://github.com/buptzzy2018/DA_MG.git
```

2. Create conda env and install the requirements

```base
conda create -n sunine python=3.9
conda activate sunine
conda install pytorch==1.12.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


## Methodologies

### Domain alignment method
+ [x] MMD
+ [x] DeepCORAL
+ [x] Center
+ [x] WBDA



## Model training steps

The model training steps follow the same procedure as described in [sunine](https://gitlab.com/csltstu/sunine). Additionally, we've introduced various multi-genre domain alignment methods such as MMD, DeepCORAL, Center Loss, and WBDA.

To use these methods:

+ [x] Modify the 'config' parameter found in 'ICASSP_2024/egs/ResNet/run.sh'.
+ [x] Detailed configurations are available in 'ICASSP_2024/egs/ResNet/conf'.

For the implemented code, refer to 'ICASSP_2024/trainer/loss'.
