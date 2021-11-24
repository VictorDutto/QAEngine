def mean_reciprocal_rank(ranks):
  quotient = [1 / rank for rank in ranks]
  result = sum(quotient) / len(quotient)
  return result

def get_ranks_for_mrr(questions_and_answers, unique_questions, I):

  ranks = []

  for i in range(len(I)):
    for rank in range(len(I[i])):
      if I[i][rank] in questions_and_answers[unique_questions[i]]:
        ranks.append(rank + 1)
        break

  return ranks