# AdCPG
Code release for the [paper](https://doi.org/10.1145/3576915.3623084) entitled "AdCPG: Classifying JavaScript Code Property Graphs with Explanations for Ad and Tracker Blocking", published at CCS 2023.

## Requirements
1. Install dependencies.
```
$ pip install adblockparser
$ pip install escodegen
$ pip install esprima
$ pip install networkx
$ pip install pandas
$ pip install requests
$ pip install scikit-learn
$ pip install selenium-wire
$ pip install torch
$ pip install torch_geometric
```

2. Install Joern by following the instructions in this [link](https://github.com/joernio/joern).

## Execution
### Dataset
JavaScript files are stored in `<AdCPG_directory>/data/scripts`.
```
$ cd <AdCPG_directory>
$ python crawler.py
```

### Phase I: Building CPGs
CPGs are stored in `<AdCPG_directory>/data/scripts`.
```
$ cd <AdCPG_directory>
$ python builder.py <Joern_directory>
```

### Phase II: Classifying CPGs
Classification results are stored in `<AdCPG_directory>/data/results`.
```
$ cd <AdCPG_directory>
$ python classifier.py
```

### Phase III: Generating Explanations
Explanations are stored in `<AdCPG_directory>/data/results/explanations`.
```
$ cd <AdCPG_directory>
$ python explainer.py
```

## Citation
```
@inproceedings{10.1145/3576915.3623084,
    author = {Lee, Changmin and Son, Sooel},
    title = {AdCPG: Classifying JavaScript Code Property Graphs with Explanations for Ad and Tracker Blocking},
    year = {2023},
    booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
    pages = {3505–3518},
}
```
