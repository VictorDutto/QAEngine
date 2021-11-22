from datasets import ClassLabel, Sequence
import random
from IPython.display import display, HTML

import pandas as pd

def show_random_elements(dataset, num_examples=10):
    '''
    Shows num_examples random elements from given datasets.

            Parameters:
                    dataset: a data of type datasets.arrow_dataset.Dataset 
                    num_examples: An integer

            Returns:
                    None
    '''
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))