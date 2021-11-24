# QAEngine


## Usage
`pip install -r requirements.txt`

A little note, this project uses faiss, which is not compatible with all python version
Thus, if you have difficulties with the installation, make sure your 
`python --version` is one of
* Python :: 2.7
* Python :: 3.5 
* Python :: 3.6
* Python :: 3.7 

`jupyter notebook`

#### authors
- alexis.morton
- clara.david
- theo.mitrail
- victor.dutto

#### Architecture description
We left each part in its respective notebook since we are not currently satisfied of the way we use our QAmodel to answer our Searchable Index entries.
Thus, part 1 is in P1 and such...

In `tools/utils.py` are the functions which aim to package the downloading of datas

In the src folder will be all the functions which can be used as a module:
`src/evaluation.py` contains the function used to evaluate predictions
`src/metrics.py` contains the function used to measure distances
`src/model_computation.py` contains the function used to compute the search index
`src/predict.py` contains the function used to predict answers
`src/preprocess.py` contains the function used to preprocess dbpedia
`src/utils.py` contains the function used to compute features and globals
`src/vizualize.py` contains the function used to vizualize entries

In the data folders are saved variables which can be loaded to avoid computations