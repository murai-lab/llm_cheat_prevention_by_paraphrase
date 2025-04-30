# Quick testing of BertScore, using distilbert specifically

import bert_score

import numpy as np


def main():
    print("Testing distilbertScore...")

    # affirmative
    options1 = ["yes", "yup", "yeah"]
    # negative
    options2 = ["no", "nope", "not"]

    # chaos
    
    scorer = bert_score.BERTScorer(model_type="distilbert-base-uncased", lang="en")

    # sent1 = "YES"
    sent1 = "NO"
    print("Sentence to compare: " + sent1)
    # Average vs options1
    P, R, F1 = scorer.score([sent1]*3, [options1])
    print("Average of affirmative options~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Precision: " + str(np.mean(np.array(P))))
    print("Recall: " + str(np.mean(np.array(R))))
    print("F1: " + str(np.mean(np.array(F1))))

    # Average vs options2
    P, R, F1 = scorer.score([sent1] * 3, [options2])
    print("Average of negative options~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Precision: " + str(np.mean(np.array(P))))
    print("Recall: " + str(np.mean(np.array(R))))
    print("F1: " + str(np.mean(np.array(F1))))


if __name__ == "__main__":
    main()