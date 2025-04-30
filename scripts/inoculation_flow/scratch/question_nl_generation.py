# Jeremy Lim
# jlim@wpi.edu

# Querying the database, generating basic questions.
import sys, os, random
import copy
import pickle

import numpy as np
import torch
from neo4j import GraphDatabase



from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, pipeline
# Note: Install is not great; need to sub in a file from this repository.
# from tokenization_small100 import SMALL100Tokenizer, FAIRSEQ_LANGUAGE_CODES
from tokenization_small100 import FAIRSEQ_LANGUAGE_CODES

from transformers import GPT2Tokenizer
from transformers import GPT2Model, GPT2LMHeadModel

# for perplexity
import evaluate


# Switching to regular m2m



TGT_LANGUAGE = "en"
TRANSLATE_TEMP = 1.3
TORCH_DEVICE = "cpu"

# pypi page: https://pypi.org/project/neo4j/
# docs: https://neo4j.com/docs/api/python-driver/current/api.html

# Resources I'm looking at to help build queries:
# https://stackoverflow.com/questions/56277210/neo4j-cypher-count-distinct-node-combinations-that-match-a-pattern
# Skip/limit idea: https://stackoverflow.com/questions/12510696/neo4j-is-there-a-way-how-to-select-random-nodes

# First query (1): Find a triplet (start, relation, end) that is unique; so if you specify a start and relation type,
#  only one node can be the answer.
# Example 1: match (n)-[r]->(m) where COUNT { MATCH(n)-[{name:r.name}]->() } = 1 return n, m limit 10
# # explicitly counts how many other relations share the same name. But needs to be modified for random sampling...
# Modified: No self-node loops: match (n)-[r]->(m) where COUNT { MATCH(n)-[{name:r.name}]->() } = 1 and n <> m return n, m limit 10
# Next step: how to query and select some efficiently at random
# Trying to subquery some nodes randomly: CALL {MATCH (n) return n order by rand() limit 1000} return n limit 1
# This conditions on addidx, which means I can pass a list of random nodes into query to make it fast:
# match (n)-[r]->(m) where COUNT { MATCH(n)-[{name:r.name}]->() } = 1 and n <> m and n.addidx < 10000 return n, m, r order by rand() limit 1


# Neat stuff to look at: https://huggingface.co/blog/constrained-beam-search

def main():

    print("Start")

    quest_seeder = SeedQuestionGenerator()

    # Testing this translation model... testing paraphrasing via backtranslation
    # https://huggingface.co/alirezamsh/small100
    # JL - modifying example code to try backtranslation.
    # model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    # tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

    # Switching to official m2m model. I suspect the tokenizer for the above is borked.
    # https://huggingface.co/facebook/m2m100_418M
    # model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    #
    # # to choose random intermediate language
    # language_options = copy.deepcopy(FAIRSEQ_LANGUAGE_CODES["m2m100"])
    # # Remove English, this is the final target language.
    # language_options.remove("en")

    # DEBUGGING: Having some weird model outputs.
    # debug_sentence = "The brown fox jumps over the lazy dog."
    #
    # debug_lang_choice = random.choice(language_options)
    #
    # debug_backtranslate = forward_back_translate(model, tokenizer, debug_sentence, debug_lang_choice)
    #
    # print("Debug sentence: " + debug_sentence)
    # print("lang choice: " + debug_lang_choice)
    # print("Backtranslated: ")
    # print(debug_backtranslate)


    qgen_model = QualityControlPipeline('questions')
    # try other options
    # qgen_model = QualityControlPipeline('sentences')
    # qgen_model = QualityControlPipeline('captions')

    # Number found when creating database.
    max_addidx = 4574679

    # Select from a set of this many nodes, then do the query on them.
    random_sample_size = 3000
    # Smaller sample size means faster run time, but increased odds of the query not finding a node that fits, even if there is one!

    sample_idxs = [random.randrange(max_addidx+1) for x in range(random_sample_size)]

    uri = "neo4j://localhost:7687"
    # Really basic local database for development. Not a production instance...
    auth = ("neo4j", "abcd1234")

    database_name = "neo4j"

    trial_count = 10

    with GraphDatabase.driver(uri, auth=auth) as driver:

        for a in range(trial_count):
            print("Trial: " + str(a))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # Simple memorization question: single node -> relationship. Logic is added to make it unambiguous, so only one answer expected.
            results = driver.execute_query(
                "match (n)-[r]->(m) where COUNT { MATCH(n)-[{name:r.name}]->() } = 1 and n <> m and n.addidx in $idxlist return n, r, m order by rand() limit 1",
                database_=database_name,
                idxlist=sample_idxs
            )
            # variable keys: n, r, m (in this order). Use the records property, then

            if len(results.records) != 0:
                start_node = results.records[0][0]
                relation = results.records[0][1]
                end_node = results.records[0][2]  # This is the answer

                # Can access properties like a dictionary...
                # concatenate results, build question "rough form"
                # start_q_string = start_node["name"] + " " + relation["name"] + "?"

                # Slighty more detailed "Prompt"?
                # start_q_string = "What is " + start_node["name"] + " " + relation["name"] + "?"

                start_q_string = quest_seeder.create_seed_memorization_question(start_node["name"], relation["name"])

                print("Seed question: ")
                print(start_q_string)
                print("Basic answer: ")
                print(end_node["name"])

                # Initial notes on question generation, after I generated a few
                # How to control Proper Nouns that must not change? Emphasize - lot of movie/media references here!
                # - Note: Some difficult questions rely on specific wording/naming to have even a chance of answering!
                # The alias issue: Some aliases are poor/weird. Some don't seem to match...
                # Incomplete knowledge: KG missing links potentially. General issue.
                # - Could lead to more answers than the KG has, potentially...
                # Also, some answers seem to be incorrect?! KG needs to be accurate!
                # Some answers a redundant: Node 1 has the word that describes node 2 already in the name...
                # Synonym issue? For example, people are instances of human, but are commonly described by occupation too.
                # -Handle "instance of" specifically?
                # Idea: Use answer string in formulation of question. It would help fluency in some cases.

                # Annoying example:
                # Basic question:
                # ivan babanovski instance of ?
                # Basic answer:
                # Huamn

                print("-----------------------")

                # intermediate_lang = random.choice(language_options)
                # print("Intermediate language: " + str(intermediate_lang))
                #
                # print("More fluent question: ")
                # paraphrase_string = forward_back_translate(model, tokenizer, start_q_string, intermediate_lang)
                # print(paraphrase_string)
                # # print("Basic answer: ")
                # # print(end_node["name"])

                print("Sampling of paraphrases: ")
                paraphrase_set = qgen_model.sample_paraphrases(start_q_string, num_paraphrases=30)

                # Try using a different context: Q + A to help paraphrase generation. Then re-extract question only?

                for idx, p in enumerate(paraphrase_set):
                    print(str(idx) + ": " + p)

                print("-----------------------")


            else:
                print("Could not find unambiguous pattern!")
                continue



    print("Done")


def sample_qa_set(num_questions=100):
    # Make a consistent sample of questions to help evaluate paraphrasing approaches.

    # Number found when creating database.
    max_addidx = 4574679

    # Select from a set of this many nodes, then do the query on them.
    random_sample_size = 3000
    # Smaller sample size means faster run time, but increased odds of the query not finding a node that fits, even if there is one!

    sample_idxs = [random.randrange(max_addidx + 1) for x in range(random_sample_size)]

    uri = "neo4j://localhost:7687"
    # Really basic local database for development. Not a production instance...
    auth = ("neo4j", "abcd1234")

    database_name = "neo4j"

    # We'll pickle the results in a database.
    questions_list = []

    with GraphDatabase.driver(uri, auth=auth) as driver:

        for a in range(num_questions):

            # Simple memorization question: single node -> relationship. Logic is added to make it unambiguous, so only one answer expected.
            results = driver.execute_query(
                "match (n)-[r]->(m) where COUNT { MATCH(n)-[{name:r.name}]->() } = 1 and n <> m and n.addidx in $idxlist return n, r, m order by rand() limit 1",
                database_=database_name,
                idxlist=sample_idxs
            )
            # variable keys: n, r, m (in this order). Use the records property, then

            if len(results.records) != 0:
                head_node = results.records[0][0]
                relation = results.records[0][1]
                tail_node = results.records[0][2]  # This is the answer

                print("Head: " + head_node['name'] + "; Relation: " + relation['name'] + "; Tail: " + tail_node['name'])

                questions_list.append({
                    'head_node': head_node['name'],
                    'relation': relation['name'],
                    'tail_node': tail_node['name']
                })




            else:
                # print("Could not find unambiguous pattern!")
                raise Exception("Could not find unambiguous pattern!")

    fname = 'paraphrase_sample_' + str(num_questions) + '.pickle'
    with open(fname, 'wb') as f:
        pickle.dump(questions_list, f)

    print("Done; 100 samples created.")



# Simple class to generate seed questions via templates.
class SeedQuestionGenerator:
    def __init__(self):
        # # DistilGPT2, developed by HuggingFace
        # self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        # self.model = GPT2LMHeadModel.from_pretrained('distilgpt2',
        #                                         device_map="cpu",
        #                                         torch_dtype=torch.float32)

        self.perplexity_module = evaluate.load("perplexity", module_type="metric")

        # Basic templates - test choosing between them
        self.simple_templates = ["Who is ",
                                 "What is ",
                                 "When is ",
                                 "Where is "]


    # Inspiration: https://huggingface.co/docs/transformers/en/perplexity
    # Also inspiration: https://stackoverflow.com/questions/75886674/how-to-compute-sentence-level-perplexity-from-hugging-face-language-models
    # https://huggingface.co/docs/evaluate/en/index
    def _get_perplexity(self, in_string):
        # tokens = self.tokenizer(in_string, return_tensors='pt')
        #
        # # Attributes:
        # # 'input_ids'
        # # 'attention_mask'
        #
        # # loop through; get each token's loss in terms of previous tokens
        #
        #
        #
        # with torch.no_grad():
        #
        #     mdl_output = self.model(tokens['input_ids'], labels=)

        results = self.perplexity_module.compute(predictions=[in_string], model_id='distilgpt2')
        print(results['perplexities'][0])

        print("Debug")


    def create_seed_memorization_question(self, h_name, r_name):
        # Testing very simple templates for now.
        templated_candidates = [x + h_name + " " + r_name + "?" for x in self.simple_templates]

        rankings = self.perplexity_module.compute(predictions=templated_candidates, model_id='distilgpt2')['perplexities']

        print(templated_candidates)
        print(rankings)

        print("Debug initial part...")
        debug_candidates = [x + h_name for x in self.simple_templates]

        debug_rankings = self.perplexity_module.compute(predictions=debug_candidates, model_id='distilgpt2')[
            'perplexities']
        print(debug_candidates)
        print(debug_rankings)

        return templated_candidates[np.argmin(rankings)]


def forward_back_translate(model, tokenizer, start_string, intermediate_lang):
    # Translate from and back to english.
    # Try to increase fluency of the selected sentence...
    tokenizer.src_lang = TGT_LANGUAGE
    encoded_1 = tokenizer(start_string, return_tensors="pt")
    # generated_tokens = model.generate(**encoded_1)
    # tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # generated_tokens = model.generate(**encoded_1, do_sample=True, temperature=TRANSLATE_TEMP)
    generated_tokens = model.generate(**encoded_1, do_sample=True, forced_bos_token_id=tokenizer.get_lang_id(intermediate_lang))


    # Note: Decode/encode required here?
    intermediate = tokenizer.decode(generated_tokens[0, :], skip_special_tokens=True)

    # print(intermediate)
    tokenizer.src_lang = intermediate_lang
    intermediate = tokenizer(intermediate, return_tensors="pt")

    # final_tokens = model.generate(**intermediate, do_sample=True, temperature=TRANSLATE_TEMP)
    final_tokens = model.generate(**intermediate, do_sample=True, forced_bos_token_id=tokenizer.get_lang_id(TGT_LANGUAGE))

    return tokenizer.decode(final_tokens[0, :], skip_special_tokens=True)


# QCPG question generation
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


    def sample_paraphrases(self, input_str, num_paraphrases=30):
        # Convenience function for getting a sample of questions given a random KG fact.

        # Better semantic, lexical quality.
        # lexical_qual = 1.0
        # syntactic_qual = 0.5
        # semantic_qual = 1.0
        # paraphrases = self(input_str, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual,
        #                         num_beams=num_paraphrases, num_beam_groups=num_paraphrases, diversity_penalty=0.8, num_return_sequences=num_paraphrases,
        #                         no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

        # Need more syntactic variation
        # lexical_qual = 0.5
        # syntactic_qual = 1.0
        # semantic_qual = 1.0
        # paraphrases = self(input_str, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual,
        #                         num_beams=num_paraphrases, num_beam_groups=num_paraphrases, diversity_penalty=0.8, num_return_sequences=num_paraphrases,
        #                         no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)


        # Syntactic variation is lacking badly. Having trouble improving it!
        # lexical_qual = 1.0
        # syntactic_qual = 1.0
        # semantic_qual = 1.0
        # paraphrases = self(input_str, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual,
        #                         num_beams=num_paraphrases, num_beam_groups=num_paraphrases, diversity_penalty=0.8, num_return_sequences=num_paraphrases,
        #                         max_new_tokens=250, top_k=50, top_p=0.95)

        # SYNTACTIC VARIATION NEEDED!
        lexical_qual = 0.5
        syntactic_qual = 0.5
        semantic_qual = 1.0
        paraphrases = self(input_str, lexical=lexical_qual, syntactic=syntactic_qual, semantic=semantic_qual,
                                num_beams=num_paraphrases, num_beam_groups=num_paraphrases, diversity_penalty=0.8, num_return_sequences=num_paraphrases,
                                max_new_tokens=250, top_k=50, top_p=0.95)


        return [x['generated_text'] for x in paraphrases]


if __name__ == "__main__":
    sample_qa_set(num_questions=100)
    main()

    # # Debugging
    # quest_seeder = SeedQuestionGenerator()
    # test_str = "Hello world!"
    # test2_str = "unquestionably pizza sky information"
    #
    # quest_seeder._get_perplexity(test_str)
    # quest_seeder._get_perplexity(test2_str)
    print("Done")