# Fine-tuning GFM model ensemble 
There are a few steps

## Installation 
### HydraGNN

[GFM](https://github.com/ORNL/HydraGNN/tree/Predictive_GFM_2024) 

### Download model ensemble 

Downlod the model ensemble from [HuggingFace](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024).
The model checkpoints and config files for each member of the ensemble are organized in a directory. 
Each member will be fine-tuned. 

## Data preparation
### Generating adios dataset
This directory contains utilities for
wrangling datasets into useful
shape for HydraGNN to operate on them.


The first step is to describe your dataset
using a `descr.yaml` file.
The format of this file is given by example:

    name: "clinical toxicity"
    authors: "Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing and Vijay Pande"
    source: https://github.com/deepchem/deepchem/tree/master/examples/clintox
    ref: https://doi.org/10.1039/C7SC02664A

    split:  null # no test/train split column present
    smiles: smiles
    graph_tasks:
      - name: FDA_APPROVED
        type: binary
        description: ""
      - name: CT_TOX
        type: binary
        description: ""

Next, import your dataset into ADIOS2 format, using

    import_csv.py --input <path_to>.csv --descr <path_to_desc>.yaml --output <path/to/output>.bp

## Fine-tuning
### Configuration
Then create a `finetune_config.json` file describing
the topology of your fine-tuning heads based
 on the tasks defined in the model. 

    yaml_to_config.py <path_to_data_description>.yaml <pretrained_config>.json <data_finetuning_config>.json

### Fine-tuning 

    python ensemble_fine_tune.py <path/to/ensemble> <data_finetuning_config>.json <dataset>.bp

