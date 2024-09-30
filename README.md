# On the alignment of LM language generation and human language comprehension

This repository contains the code to reproduce the experiments of the paper "On the alignment of LM language generation and human language comprehension". 

## Setup and requirements

The code is based on the PyTorch and huggingface modules.
```bash
pip install -r requirements.txt
```

## Download the data and set path names

* download the EMTeC eye-tracking data via [https://github.com/DiLi-Lab/EMTeC](https://github.com/DiLi-Lab/EMTeC) with the `get_et_data.py` script
* make sure the eye-tracking data files are in a directory called `data`:
```
├── data
    ├── stimuli.csv
    ├── reading_measures_corrected.csv
```
* download the EMTeC transition score tensors via [https://github.com/DiLi-Lab/EMTeC](https://github.com/DiLi-Lab/EMTeC) with the `get_tensors.py` script
* indicate the path to the tensors in the script `CONSTANTS.yaml`


## Extract surprisal and entropy from the raw transition scores

Compute surprisal and contextual entropy from the LLMs' transition scores and merge them with the file containing the reading measures from EMTeC.
```bash
bash extract_scores.sh
```

## Estimate contextual entropy with the LLMs

Estimate contextual entropy with the LLMs used for the stimulus generation in EMTeC and merge them with the reading measures and surprisal values. 

**Note:** In order to prompt the models, you need GPUs set up with [CUDA](https://developer.nvidia.com/cuda-downloads). 

Beware that the GPUs are hard-coded in the bash script and depending on the kind of GPUs available, please 
adapt them accordingly.
```
bash extract_entropy.sh
```

## Run the analyses

* our analyses are implemented in R
* some of our models are implemented in Julia. Please download Julia from [https://julialang.org/downloads/](https://julialang.org/downloads/)
* indicate the path to Julia (e.g., `Users/username/.juliaup/bin`) in `CONSTANTS.yaml`


Run the analysis script:
```bash
Rscript analyses.R
```


## Citation

```bibtex
@inproceedings{bolliger2024alignment,
    title = {On the alignment of LM language generation and human language comprehension},
    author = {Bolliger, Lena S. and Haller, Patrick and J{\"a}ger, Lena A.},
    booktitle = {Proceedings of the 7th {BlackboxNLP} workshop: {A}nalysing and interpreting neural networks for NLP},
    month = {nov},
    year = {2024},
    address = {Miami},
    publisher = {Association for Computational Linguistics},
}
```