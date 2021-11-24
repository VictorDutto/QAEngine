from sentence_transformers import SentenceTransformer
import sentence_transformers.util
import time
import numpy as np
import faiss

from src.metrics import mean_reciprocal_rank


def sentence_transformer_tests(model, is_dot, df, unique_questions, q_and_a):
    start = time.time()

    embeddings = model.encode(df.context.to_list(), show_progress_bar=True)
    questions_embedding = model.encode(unique_questions, show_progress_bar=True)

    if is_dot:
        res = sentence_transformers.util.dot_score(questions_embedding, embeddings)
    else:
        res = sentence_transformers.util.pytorch_cos_sim(questions_embedding, embeddings)

    ranks = []

    for i in range(len(res)):
        similarities = res[i]
        list_of_similarities_sorted_in_reverse = sorted(range(len(similarities)), key = lambda x: similarities[x])
        list_of_similarities_sorted_in_reverse.reverse()
        question = unique_questions[i]
        for rank in range(len(list_of_similarities_sorted_in_reverse)):
            if list_of_similarities_sorted_in_reverse[rank] in q_and_a[question]:
                ranks.append(rank + 1)
                break


    result = mean_reciprocal_rank(ranks)
    end = time.time()

    run_time = end - start

    return (result, embeddings, questions_embedding, run_time)


def k_nearest_neighbours_context(question, k_number_of_results, df):
  model = SentenceTransformer('msmarco-distilbert-base-tas-b')
  embeddings = model.encode(df.context.to_list(), show_progress_bar=True)
  
  _, index = get_embeddings_and_index(embeddings, df)
  return k_nearest_neighbours_context_(model, question, k_number_of_results, df, index)
 

def k_nearest_neighbours_context_(model, question, k_number_of_results, df, index):
  list_k_context = []
  D, I = vector_search([question], model, index, num_results=k_number_of_results)
  list_of_all_context = df.context.to_list()
  ids = I[0]
  for context_id in ids:
    list_k_context.append(list_of_all_context[context_id])
  return list_k_context


def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level
    DistilBERT model and finds similar vectors using FAISS.
    
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

def get_embeddings_and_index(embeddings, df):
  # Step 1: Change data type
  embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

  # Step 2: Instantiate the index
  index = faiss.IndexFlatL2(embeddings.shape[1])

  # Step 3: Pass the index to IndexIDMap
  index = faiss.IndexIDMap(index)

  # Step 4: Add vectors and their IDs
  index.add_with_ids(embeddings, df.index.values)

  return (embeddings, index)