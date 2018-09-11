# some useful functions
# not related to this project

from rouge import Rouge
from nltk.translate import bleu_score
from functools import reduce

def generation_metrics(hypothesis_list, reference_list):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(
        hyps=list(reduce(lambda x,y:x+y,list(zip(*([hypothesis_list]*len(reference_list)))))),
        refs=reference_list * len(hypothesis_list),
        avg=True)
    bleu_scores = {}
    for i in range(1,6):
        bleu_scores['bleu%s'%i] = bleu_score.corpus_bleu(
            list_of_references=[reference_list]*len(hypothesis_list),
            hypotheses=hypothesis_list,
            weights=[1.0/i]*i)
    return rouge_scores, bleu_scores