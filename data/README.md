# Manual Evaluation Data

Provided in the manual_evals are the manual evaluation results for this project. The author has manually annotated whether a question + answer pair was a correctly generated paraphrase of the original MMLU question.

A paraphrased question has the same semantic meaning and answer as the original question, and uses the same subject-matter jargon. In other words, a paraphrase that changes jargon to similar sounding wording is also considered an incorrect paraphrase.

| Filename                      | No. Evaluations | No. Annotated by Author as Correct Paraphrase | Notes                                                                                    |
|-------------------------------|-----------------|---------------------------------------------|------------------------------------------------------------------------------------------|
| eval_all.[csv/pickle]         | 200             | 121                                         | Balanced sampling of all MMLU subjects evaluated, before filtering.                      |
| eval_all_correct.[csv/pickle] | 101             | 27                                          | Balanced sampling of final candidate paraphrases, after filtering.                       |
| eval_alg.[csv/pickle]         | 33              | 5                                           | Evaluations for abstract algebra subject, examined for outlier results.                  |
| eval_alg.[csv/pickle]         | 50              | 10                                          | Evaluations for a sampling of the moral scenarios subject, examined for outlier results. |

# Building MMLU Dataset

You may pull a copy of MMLU down from [here](https://github.com/hendrycks/test)

Extract the data.tar into the data/mmlu_data:

```shell
mkdir data/mmlu_data/ # Create if not exists
tar -xvf data.tar -C mmlu_data/
```

# Building PAWS Dataset

You may follow these instructions for building the PAWS dataset from [here](https://github.com/google-research-datasets/paws)

PAWS has 2 sub-datasets: PAWS-WIKI and PAWS QQP.

## PAWS-Wiki

PAWS-Wiki can be directly downloaded [here](https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz).

Please install it in data/PAWS_Wiki:
```shell
mkdir data/PAWS_Wiki/ # Create if not exists
cd data/PAWS_Wiki/
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
tar -xzvf paws_wiki_labeled_final.tar.gz
```

## PAWS-QQP

This is more involved due to license restrictions. Follow the instructions on the PAWS repository to generate the PAWS-QQP dataset with their provided scripts, but install the dataset in data/PAWS_QQP/

```shell
mkdir data/PAWS_QQP/ # Create if not exists
# Copy the data into this directory after following instructions on PAWS repository.
```