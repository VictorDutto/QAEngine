from datasets import load_metric
import collections

def evaluate_predictions(final_predictions, datasets, squad_v2 = True) -> dict :
    '''
    Evaluate predictions from a model

            Parameters:
                    final_predictions: A collections.OrderedDict
                    datasets : A datasets.dataset_dict
                    squad_v2 : A boolean
                    

            Returns:
                    a dict of several numerical scores
    '''
    metric = load_metric("squad_v2" if squad_v2 else "squad") 
    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    return metric.compute(predictions=formatted_predictions, references=references)