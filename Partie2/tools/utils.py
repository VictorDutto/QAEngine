from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset

def download_dbpedia():
    dataset = "dbpedia"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def download_squadv2():
    return load_dataset("squad_v2")