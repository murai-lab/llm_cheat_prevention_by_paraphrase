# Jeremy Lim
# Testing QCPG for generating paraphrases. Similar to tiny_llama_example

import sys, os

import numpy as np
import torch
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizerFast

# Provided copypasta: # https://github.com/IBM/quality-controlled-paraphrase-generation
class QualityControlPipeline:

    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
            f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control = {name: max(min(val, self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in
                   zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)

# end provided copypasta

def main():
    print("Start")

    # test_question = '\"What is the distance from Mars to the Sun when it is at aphelion?\"'

    # very explicitly
    test_question = 'What is the distance from Mars to the Sun when Mars is at aphelion?'

    # github page; they provide code to copypasta...
    # https://github.com/IBM/quality-controlled-paraphrase-generation

    gen_model = QualityControlPipeline('questions')

    # generate a paraphrase? Uses a T5 model. Hopefully it will fit...
    # Control values: Higher means better quality.
    # High lexical - more lexical variation
    # High syntactic - more syntactic variation
    # High semantic - better semantic matching with original.

    # lexical_qual = 0.5
    # syntactic_qual = 0.5
    # semantic_qual = 1.0  # emphasize semantic similarity; most important!
    # The class uses huggingface pipeline kwargs, so docs for reference:
    # https://huggingface.co/docs/transformers/en/main_classes/pipelines

    print("Test 30 paraphrases:")
    # Using some tested parameters from tinyllama
    # Doesn't understand the word "aphelion", mixes it up with "apollo" and "lion" or "ion"
    # also: Misspellings! Extremely Annoying...
    # lexical_qual = 0.5
    # syntactic_qual = 0.5
    # semantic_qual = 1.0  # emphasize semantic similarity; most important!
    # paraphrases = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=5, diversity_penalty=0.8, num_return_sequences=30,
    #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

    # Allow high syntactic quality, low lexical quality?
    # lexical_qual = 0.1
    # syntactic_qual = 1.0
    # semantic_qual = 1.0  # emphasize semantic similarity; most important!
    # paraphrases = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=5, diversity_penalty=0.8, num_return_sequences=30,
    #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

    # Expansive beam search. Slightly improve lexical quality.
    # lexical_qual = 0.3
    # syntactic_qual = 1.0
    # semantic_qual = 1.0  # emphasize semantic similarity; most important!
    # paraphrases = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=30, diversity_penalty=0.8, num_return_sequences=30,
    #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

    # What happens when semantic quality is low?
    # lexical_qual = 0.5
    # syntactic_qual = 0.5
    # semantic_qual = 0.1
    # paraphrases = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=30, diversity_penalty=0.8, num_return_sequences=30,
    #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

    # Better semantic, lexical quality.
    lexical_qual = 1.0
    syntactic_qual = 0.5
    semantic_qual = 1.0
    paraphrases = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=30, diversity_penalty=0.8, num_return_sequences=30,
                                    no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

    for idx, p in enumerate(paraphrases):
        print(str(idx) + ": " + p['generated_text'])

    # print("test 10 paraphrases")
    # for a in range(10):
    #     # Using some tested parameters
    #     paraphrase = gen_model(test_question, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual, num_beams=30, num_beam_groups=5, diversity_penalty=0.8, num_return_sequences=30,
    #                                     no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)
    #     print(str(a) + " : " + paraphrase[0]['generated_text'])


    print("Done")


if __name__ == "__main__":
    main()