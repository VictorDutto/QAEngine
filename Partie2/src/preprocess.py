import pandas as pd

def size_context(context):
    context_tokens = context.split(' ')
    context_words = [word.lower() for word in context_tokens if word != '']
    return context_words

def select_only_50words(corpus, queries, qrels):
    words50 = []

    for qrel in qrels:
        
        if "QALD2" in qrel:
            question = queries[qrel]
            
            for elt in qrels[qrel]:
                title = corpus[elt]['title']
                context = corpus[elt]['text']
                
                answer = (corpus[elt]['title'], corpus[elt]['text'])
                
                if len(size_context(context)) >= 50:
                    res = [context, title, question, answer]
                    words50.append(res)

    return words50


def create_dataframe_from_squad_and_dbdebia(datasets, words50):
    count = 0
    all_context = []
    dico = dict()

    for i in range(len(datasets['validation'])):
        context = datasets['validation'][i]['context']
        if context not in dico:
            new_dico = {}
            new_dico['title'] = datasets['validation'][i]['title']
            new_dico['question'] = datasets['validation'][i]['question']
            all_context.append(datasets['validation'][i]['question'])
            new_dico['answers'] = datasets['validation'][i]['answers']
            dico[context] = new_dico

    for i in range(len(words50)):
        if words50[i][0] not in dico:
            all_context.append(words50[i][0])
            new_dico = {}
            new_dico['title'] = ""
            new_dico['question'] = ""
            new_dico['answers'] = ""
            dico[words50[i][0]] = new_dico

    questions = []
    contexts = []
    titles = []
    answers = []

    for context in dico.keys():
        contexts.append(context)
        questions.append(dico[context]['question'])
        titles.append(dico[context]['title'])
        answers.append(dico[context]['answers'])

    df = pd.DataFrame(list(zip(questions, contexts, titles, answers)), 
                    columns =['question', 'context', 'title', 'answers'])

    return (df, questions, contexts, titles, answers)


def create_unique_questions(questions_list):
    seen = set()
    unique_questions = [x for x in questions_list if not (x in seen or seen.add(x))]
    return unique_questions

def create_question_and_answer(df, unique_questions):
    question_and_answer = dict()

    for question in unique_questions:
        a = df.loc[df['question'] == question]
        l = list(a.index)
        question_and_answer[question] = l

    return question_and_answer