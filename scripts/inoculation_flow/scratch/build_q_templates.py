# Jeremy Lim
# Use SQuAD to build Q/A templates

import sys, os
import json
import subprocess
import time
import pickle
import random

import nltk
from nltk.parse import CoreNLPParser
from nltk import Tree

# load tagset info
# nltk.download('tagsets_json')

# NLTK corpus sources: http://www.nltk.org/nltk_data/
# Use NLTK to download corpus: https://www.nltk.org/api/nltk.downloader.html
# nltk.download("brown")

import torch
import numpy as np

# Link: https://rajpurkar.github.io/SQuAD-explorer/

# Load SQuAD data, build templates.

# Wrapper code for using BERT to decide between templates
import BERTMLMWrapper
# TODO: Find sensible alpha for multi-token discount

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM

from transformers import RobertaTokenizer, RobertaForMaskedLM

from transformers import DebertaTokenizer, DebertaForMaskedLM

# Other notes:
# Installing java: https://ubuntu.com/tutorials/install-jre#2-installing-openjdk-jre
# Downloading latest version: https://stanfordnlp.github.io/CoreNLP/

# Running corenlp server locally:
# https://stanfordnlp.github.io/CoreNLP/cmdline.html
# https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

# Command from the github page:
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
# -status_port 9000 -port 9000 -timeout 15000 &

# Will try to use python subprocess: https://docs.python.org/3/library/subprocess.html
# Helpful: https://www.datacamp.com/tutorial/python-subprocess
# Subprocess cwd: https://stackoverflow.com/questions/21406887/subprocess-changing-directory

def parse_all():

    data_path = "/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/SQuAD_Data/train-v2.0.json"
    corenlp_server_path = "/home/jeremy/Documents/WPI_MS/CoreNLP_Server/stanford-corenlp-4.5.7"

    parsed_qs_fname = "Parsed_SQuAD.pickle"

    with open(data_path, 'r') as f:
        SQuAD_data = json.load(f)
    # JL schema notes:
    # Under ['data'], a list of dictionaries. Each has a 'title' and some 'paragraphs'. Each 'paragraphs' is a list of dicts, with a 'context'
    # paragraph key (a string), and another 'qas' key. 'qas' is a list of dicts with a 'question' key as a string.

    # Extract questions only. We don't care about the answers in this case
    question_list = []
    for theme in SQuAD_data['data']:
        for paragraph in theme['paragraphs']:
            for question in paragraph['qas']:
                question_list.append(question['question'])

    # This is used by various papers to derive parse trees:
    # https://stanfordnlp.github.io/CoreNLP/
    # This has an interface to the above: https://stanfordnlp.github.io/stanza/
    # use in nltk: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK

    print("Number of questions to parse: " + str(len(question_list)))
    parsed_set = []

    print("Starting Parser Process...")
    # Start subprocess
    # Removed & at the end to prevent backgrounding and making python unable to talk to the process...
    # Default timeout is 15000, which I think is 15 seconds
    # increasing to 2 minutes = 120000
    corenlp_command = "java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 120000"

    # List version
    # corenlp_command = ["java", "-mx4g", "-cp \"*\"", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-preload", "tokenize,ssplit,pos,lemma,ner,parse,depparse", "-status_port", "9000", "-port", "9000", "-timeout", "15000"]

    # test
    # corenlp_command = ["java", "-version"]

    # corenlp_command_seq = []

    # Issue: Because I was using shell, the shell process was getting termination signals, not the server process!
    # Some clever tricks:
    # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
    # Trying the exec trick. Only for linux though
    nlp_process = subprocess.Popen("exec " + corenlp_command, cwd=corenlp_server_path,
                                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # shell = False



    try:
        time.sleep(10)  # wait a little before trying to connect. Avoid bugs!
        # TODO: Instead, wait for stdin from server process indicating ready. Complicated though...

        # NLTK examples; check if it's working:
        # Lexical Parser
        parser = CoreNLPParser(url='http://localhost:9000')

        print("Parser Example Outputs: ")

        # Parse tokenized text.
        # print(list(parser.parse('What is the airspeed of an unladen swallow ?'.split())))
        # Example output:
        # [Tree('ROOT', [Tree('SBARQ', [Tree('WHNP', [Tree('WP', ['What'])]), Tree('SQ', [Tree('VBZ', ['is']), Tree('NP', [
        #     Tree('NP', [Tree('DT', ['the']), Tree('NN', ['airspeed'])]),
        #     Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['an']), Tree('JJ', ['unladen'])])]),
        #     Tree('S', [Tree('VP', [Tree('VB', ['swallow'])])])])]), Tree('.', ['?'])])])]

        # Parse raw string.
        # print(list(parser.raw_parse('What is the airspeed of an unladen swallow ?')))
        # Example output:
        # [Tree('ROOT', [Tree('SBARQ', [Tree('WHNP', [Tree('WP', ['What'])]), Tree('SQ', [Tree('VBZ', ['is']), Tree('NP', [
        #     Tree('NP', [Tree('DT', ['the']), Tree('NN', ['airspeed'])]),
        #     Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['an']), Tree('JJ', ['unladen'])])]),
        #     Tree('S', [Tree('VP', [Tree('VB', ['swallow'])])])])]), Tree('.', ['?'])])])]

        # It is efficient to send many sentences at once to the server, instead of one at a time!
        chunk_size = 1000

        # test timing
        # print("Chunk start")
        # parsed_chunk = list(parser.raw_parse_sents(question_list[:1000]))
        # print("Chunk end")
        # Chunk only took 21 seconds for 1000 entries. About 1/5th of the time....

        # iterate through and parse entire question set
        # for idx, q in enumerate(question_list):
        #     if idx % 1000 == 0:
        #         print(str(idx) + "/" + str(len(question_list)))
        #
        #     # parsed_set.append(list(parser.raw_parse(q)))
        #     # 73 seconds for 1000
        #
        #
        #     # Pickle it every so often.
        #     with open(parsed_qs_fname, 'wb') as f:
        #         pickle.dump(parsed_set, f)

        # iterate through question set in chunks.
        chunk_num = int(len(question_list) / chunk_size)
        if len(question_list) % chunk_size != 0:
            chunk_num += 1

        for i in range(chunk_num):
            print(str(i*chunk_size) + "/" + str(len(question_list)))
            question_set = question_list[i:(i+chunk_size)]
            parsed_chunk = list(parser.raw_parse_sents(question_set))



            # iterate all iterators
            parsed_chunk = [list(x) for x in parsed_chunk]

            # DEBUGGING: display random one.
            rand_indx = random.choice(range(len(question_set)))
            print("Sentence: ")
            print(question_set[rand_indx])
            print_tree = parsed_chunk[rand_indx][0]

            print("Parse Tree: ")
            print_tree.pretty_print()

            print("After Substitutions: ")
            substitutions = {
                "NP": ("SUB_NP", ["{NP}"]),  # Tuple is (new_label, new_value)
                "WHNP": ("SUB_WHNP", ["{WHNP}"]),
                "VP": ("SUB_VP", ["{VP}"]),
            }
            sub_counts = substitute_nodes(print_tree, substitutions)

            print_tree.pretty_print()

            print("Template: ")
            print(" ".join(print_tree.leaves()))

            parsed_set = parsed_set + parsed_chunk

            # Pickle it every so often.
            with open(parsed_qs_fname, 'wb') as f:
                pickle.dump(parsed_set, f)


            # for a in range(30):
            #     print(parsed_set[a])

            # print("Break")

        # Understanding parts of speech tags: https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf
        # https://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        # https://courses.grainger.illinois.edu/cs447/fa2018/Slides/Lecture05.pdf
        # https://gist.github.com/nlothian/9240750
    finally:
        print("Ending Parser Process...")
        # Done with parsing
        nlp_process.terminate()  # SIGTERM
        print("Break")
        nlp_process.wait()  # Wait for termination

    with open(parsed_qs_fname, 'wb') as f:
        pickle.dump(parsed_set, f)

    print("Done")


def substitute_nodes(tree, substitution_dict):
    # Substitute some nodes of a tree, replace entire subtrees with a single leaf node
    # This is for template creation
    # Count the number of substitutions done for each substitution as well.

    sub_count = {}
    for key in substitution_dict.keys():
        sub_count[key] = 0

    # traverse tree
    # TODO: Refactor
    if type(tree) is nltk.Tree:
        # check/replace children

        # Tree handles children like a list
        for idx in range(len(tree)):
            if type(tree[idx]) is nltk.Tree:
                label = tree[idx].label()
                if tree[idx].label() in substitution_dict:
                    # replace
                    newleaf = nltk.Tree(substitution_dict[label][0], substitution_dict[label][1])
                    tree[idx] = newleaf
                    # Remember # of substitutions for that.
                    sub_count[label] += 1
                else:
                    # otherwise search.
                    sub_sub_count = substitute_nodes(tree[idx], substitution_dict)
                    # combine
                    for key in substitution_dict.keys():
                        sub_count[key] += sub_sub_count[key]

    return sub_count



def build_templates():
    # Building templates using this assumption:
    # For some (h, r, t):
    # head and tail are always objects.
    # r can be an object, or it could be a verb!

    # The built templates have to reflect this. Multiple combos may be possible...

    # restriction:
    # only 2 template substitution places.
    # At least one NP (for head), and another NP, VP.
    # Order can be switched though for two NPs.
    # Choose combos below NP/VP level?

    # NOTE: consider adjective phrase (ADJP?) for relation candidate?

    # Working with NLTK trees: https://www.nltk.org/_modules/nltk/tree.html
    # https://www.nltk.org/howto/tree.html
    # https://www.nltk.org/api/nltk.tree.html#nltk.tree.Tree.productions

    parsed_qs_fname = "Parsed_SQuAD.pickle"

    # custom labels, used when substituting/removing parts of the tree.
    # Only substituting NP/VP right now. # Tags to understand: WHNP; some question NP

    substitutions = {
        "NP": ("SUB_NP", ["{NP}"]),  # Tuple is (new_label, new_value)
        "WHNP": ("SUB_WHNP", ["{WHNP}"]),
        "VP": ("SUB_VP", ["{VP}"]),
    }
    # Notes:
    # WHNP: How to preserve initial "wh" word, but keep the rest?
    # Somehow, is there a variant of {NP} {VP} that can be replaced by a single NP?/Object?
    # SQ: Preserve initial "Did" verb?
    # Impossible grammar: "award received"? Nope, if "Bob" is head -> "What was the award recieved by Bob"

    # Two important ideas:
    # 1) Restrict BeRT to filler/preposition/articles/etc. Basically, search for a good seed question, restrict chance to change meaning.
    # 2) More context is necessary for more exotic question variations. With just h, r, it is difficult to create a lot of variety in language/wording.
    # 3) Because of 2, word-level synonym replacement might be necessary for questions that are too simple...

    with open(parsed_qs_fname, 'rb') as f:
        parse_list = pickle.load(f)

    # squeeze
    parse_list = [x[0] for x in parse_list]

    random.shuffle(parse_list)

    template_strs = []
    for p_tree in parse_list:
        # find NP/VP at the highest parts of the tree, replace them
        # Using custom label
        # orig = p_tree.copy(deep=True)
        # p_tree.pretty_print()

        sub_counts = substitute_nodes(p_tree, substitutions)

        # p_tree.pretty_print()

        total = sub_counts["NP"] + sub_counts["VP"]

        # TODO: Playing with these restrictions.
        if total >= 2:  # Only up to this many allowed.

            # if sub_counts["NP"] > 0: # At least one NP is required for head.
            # tree to string.
            # Join all of the leaves, separate by spaces
            template_strs.append(" ".join(p_tree.leaves()))
            # print(template_strs[-1])




    dedup_dict = {}
    # deduplicate.
    for candidate in template_strs:
        if candidate not in dedup_dict:
            dedup_dict[candidate] = 1
        else:
            dedup_dict[candidate] += 1

    deduped_list = [x for x in dedup_dict.keys()]


    for template in deduped_list:
        print(template)

    # JL Notes:
    # All are traditional question form. Interestingly, no "command" forms. Example:
    # When was George Washington Born? "vs"
    # Tell me when George Washington was Born.

    print("Number of basic templates created: " + str(len(template_strs)))
    print("Number of unique basic templates created: " + str(len(deduped_list)))

    print("Done")



def determine_bert_tokens():
    # use the brown corpus along with POS tagging, determine words to restrict bert to based on the tags
    brown_path = "/home/jeremy/nltk_data/corpora/brown"

    # Wikipedia actually has a decent list of the tags used: https://en.wikipedia.org/wiki/Brown_Corpus
    # Brown manual: http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM

    # A word is included if it had been tagged with one of these at least once
    # this is built from the manual: http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM
    include_tags = set(['at', # Articles
                    'do', 'dod', 'dod', # do, did, does. Special verb
                    'in', # Prepositions.
                    'be', 'bed', 'bedz', 'beg', 'bem', 'ben', 'ber', 'bez', # be variants. Special verb
                    'to', # To
                    'wdt', 'wp$', 'wpo', 'wps', 'wql', 'wrb' # All variants of "wh" words.
                    'dt', 'dti', 'dts', 'dtx',  # Determiners.
                    'ex',  # Existential there.
                    ])
    # Note: Merged tags/contractions?

    files = os.listdir(brown_path)
    # Parse files, ignore some helper files
    files = [os.path.join(brown_path, x) for x in files if "README" not in x and "CONTENTS" not in x and "cats" not in x]
    print("Number of files to parse: " + str(len(files)))

    # maintain unique set of words.
    vocab_set = set()

    for idx, fpath in enumerate(files):
        with open(fpath, 'r') as f:
            for line in f:
                # all tokens appear to be space separated. All words appear to be the format of {word}/tag
                parse_line = line.strip("\t\n ")
                if len(parse_line) <= 0:
                    continue  # Ignore empty lines

                parse_line = parse_line.split(' ')
                for tag in parse_line:
                    components = tag.split('/')
                    if components[1] in include_tags:  # if the tag matches.
                        vocab_set.add(components[0].lower())  # Let's lowercase it for now.

        if (idx+1) % 100 == 0:
            print(str(idx+1) + "/" + str(len(files)))


    # remove punctuation causing strange issues
    remove_set = ['-', ':', '.']
    for r in remove_set:
        if r in vocab_set:
            vocab_set.remove(r)

    print("Size of vocabulary built: " + str(len(vocab_set)))

    for p in vocab_set:
        print(p)


    # JL Notes:
    # Some exotic/old phrasing is making its way into this. However, I think BERT can probably filter that...

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # roberta
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # deberta
    # tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

    # Determine the token idxs for each of the vocab words.
    keep_idxs = set()
    for word in vocab_set:
        tokens = tokenizer(word)
        # strip start/end tokens
        tokens = tokens[1:-1]

        # Exclude multi token words for now as well
        if len(tokens['input_ids']) > 1:
            continue

        # Add all classes, even in the case of multiple tokens (for now).
        for idx in tokens['input_ids']:
            keep_idxs.add(idx)

    print("Number of tokens kept: " + str(len(keep_idxs)))
    # print("Done")

    return keep_idxs

def bert_syntaxify_test(max_mask_num=5):

    # JL notes:
    # Still seems to "avoid" good sentences, even for this example! So BeRT models may not have been trained on grammatical english.

    token_discount_alpha = 0.8

    example_head = "George Washington"
    example_relation = "date of birth"

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    # Real bert
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # Switching to roberta:
    # https://huggingface.co/FacebookAI/roberta-base
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # model = RobertaForMaskedLM.from_pretrained('roberta-base')

    # Try deberta:
    # https://huggingface.co/microsoft/deberta-base
    # https://huggingface.co/docs/transformers/en/model_doc/deberta
    # tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    # model = DebertaForMaskedLM.from_pretrained('microsoft/deberta-base')

    # nltk.help.upenn_tagset()
    # Tagset info: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    # Penn treebank official page: https://catalog.ldc.upenn.edu/docs/LDC95T7/cl93.html

    # number of combos: (max_mask_num +1)^3 * 2
    # For example, minimal: "h r?"
    # Maximal example, for max_mask_num=3: "[MASK] [MASK] [MASK] h [MASK] [MASK] [MASK] r [MASK] [MASK] [MASK]?"

    # For max_mask_num = 3, that's 128 evals.

    # get token lengths for head, tail
    head_tokens = tokenizer(example_head)
    relation_tokens = tokenizer(example_relation)
    punctuation_tokens = tokenizer("?")

    head_ids = head_tokens['input_ids'][1:-1]
    relation_ids = relation_tokens['input_ids'][1:-1]
    punctuation_ids = punctuation_tokens['input_ids'][1:-1]

    head_token_count = len(head_ids)
    relation_token_count = len(relation_ids)


    print("Building vocab mask...")
    num_output_logits = 30522  # this appears to be the # of logits a bertlike model outputs.
    # num_output_logits = 50265  # roberta, deberta has more output possibilities than bert
    # num_output_logits = 30522  # deberta's vocab size
    mask_arr = np.zeros(num_output_logits)
    keep_idxs = determine_bert_tokens()
    for keep in keep_idxs:
        mask_arr[keep] = 1.0

    print("Vocab mask built.")

    candidate_seeds = []

    min_mask_num = 0
    # build combos
    eval_count = 0
    for pre_masks in range(min_mask_num, max_mask_num+1):
        for middle_masks in range(min_mask_num, max_mask_num+1):
            for post_masks in range(min_mask_num, max_mask_num+1):
                print(str(eval_count) + "/" + str(2*pow(max_mask_num+1, 3)))
                # eval {masks} h {masks} r {masks}?

                # FORWARD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                combo1 = ("[MASK] " * pre_masks) + example_head + (" [MASK]" * middle_masks) + " " + example_relation + (" [MASK]" * post_masks) + "?"
                print(combo1)

                with ((torch.no_grad())):
                    combo1_encode = tokenizer(combo1, return_tensors='pt')

                    # Logits only.
                    output = torch.squeeze(model(**combo1_encode).logits)
                    # ignore start/end tokens
                    output = output[1:-1]

                    # Greedy: Choose tokens that maximize the "probability"; can try other search methods in the future.
                    pre_scores = torch.max(output[:pre_masks] * mask_arr, dim=1)
                    # Index only the fixed words for head scores
                    head_scores = output[range(pre_masks, pre_masks + len(head_ids)), head_ids]
                    middle_scores = torch.max(output[(pre_masks+head_token_count):(pre_masks+head_token_count) + middle_masks] * mask_arr,dim=1)
                    # Index only the fixed words for relation scores
                    relation_scores = output[range((pre_masks+head_token_count+middle_masks), (pre_masks+head_token_count+middle_masks) + len(relation_ids)), relation_ids]
                    end_scores = torch.max(output[(pre_masks+head_token_count+middle_masks+relation_token_count):-1] * mask_arr, dim=1)
                    punctuation_scores = output[-1, punctuation_ids[0]]

                    # sum all scores. Then compute logsumexp to normalize.
                    # Note: Logsumexp trick? https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html

                    # Debugging score issues

                    # test_first_score = torch.exp(pre_scores.values[0] -torch.sum(torch.logsumexp(output[:1], dim=1)))

                    logsum_term = torch.sum(torch.logsumexp(output, dim=1))

                    score = torch.sum(pre_scores.values) + \
                            torch.sum(head_scores) + \
                            torch.sum(middle_scores.values) + \
                            torch.sum(relation_scores) + \
                            torch.sum(end_scores.values) + \
                            torch.sum(punctuation_scores) - \
                            logsum_term

                    # Get back into probability world. Although this is unnecessary for ranking relative candidates.
                    score = torch.exp(score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha)))

                    # Decode output sentence. Include start/end tokens.
                    token_set = pre_scores.indices.tolist() + head_ids + middle_scores.indices.tolist() + relation_ids + end_scores.indices.tolist() + punctuation_ids

                    candidate_str = tokenizer.decode(token_set)

                    candidate_seeds.append((candidate_str, score))


                # REVERSE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # eval the reverse: {masks} r {masks} h {masks}?
                combo2 = ("[MASK] " * pre_masks) + example_relation + (" [MASK]" * middle_masks) + " " + example_head + (" [MASK]" * post_masks) + "?"
                print(combo2)

                with ((torch.no_grad())):
                    combo2_encode = tokenizer(combo2, return_tensors='pt')

                    # Logits only.
                    output = torch.squeeze(model(**combo2_encode).logits)
                    # ignore start/end tokens
                    output = output[1:-1]

                    # Greedy: Choose tokens that maximize the "probability"; can try other search methods in the future.
                    pre_scores = torch.max(output[:pre_masks] * mask_arr, dim=1)
                    # Index only the fixed words for head scores
                    relation_scores = output[range(pre_masks, pre_masks + len(relation_ids)), relation_ids]
                    middle_scores = torch.max(
                        output[(pre_masks + relation_token_count):(pre_masks + relation_token_count) + middle_masks] * mask_arr,
                        dim=1)
                    # Index only the fixed words for relation scores
                    head_scores = output[range((pre_masks + relation_token_count + middle_masks),
                                                   (pre_masks + relation_token_count + middle_masks) + len(
                                                       head_ids)), head_ids]
                    end_scores = torch.max(
                        output[(pre_masks + relation_token_count + middle_masks + head_token_count):-1] * mask_arr,
                        dim=1)
                    punctuation_scores = output[-1, punctuation_ids[0]]

                    # sum all scores. Then compute logsumexp to normalize.
                    # Note: Logsumexp trick? https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html

                    # Debugging score issues

                    # test_first_score = torch.exp(pre_scores.values[0] -torch.sum(torch.logsumexp(output[:1], dim=1)))

                    logsum_term = torch.sum(torch.logsumexp(output, dim=1))

                    score = torch.sum(pre_scores.values) + \
                            torch.sum(relation_scores) + \
                            torch.sum(middle_scores.values) + \
                            torch.sum(head_scores) + \
                            torch.sum(end_scores.values) + \
                            torch.sum(punctuation_scores) - \
                            logsum_term

                    # Get back into probability world. Although this is unnecessary for ranking relative candidates.
                    score = torch.exp(score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha)))

                    # Decode output sentence. Include start/end tokens.
                    token_set = pre_scores.indices.tolist() + relation_ids + middle_scores.indices.tolist() + head_ids + end_scores.indices.tolist() + punctuation_ids


                    # Potential issue: https://github.com/huggingface/transformers/issues/14502
                    # I've seen this...
                    candidate_str = tokenizer.decode(token_set)
                    candidate_seeds.append((candidate_str, score))


                    # if pre_masks == 3 and middle_masks == 1 and post_masks == 0:
                    #     print("break")
                    #     # TODO: more in depth debugging? A non-grammatical string is preferred!
                    #     # "what was the date of birth of george washington?" is an easy, grammatical seed.

                eval_count += 2

    priority_candidates = sorted(candidate_seeds, key=lambda x: x[1].item())

    for p in priority_candidates:
        print(p[0] + " ; " + str(p[1].item()))

    print("Sanity: Try some human-generated strings")

    test_1 = "George Washington had what date of birth?"
    test_2 = "When was the date of birth of George Washington?"
    print(test_1)
    print("Score 1: " + str(score_string_odds(model, tokenizer, test_1, token_discount_alpha)))

    print(test_2)
    print("Score 2: " + str(score_string_odds(model, tokenizer, test_2, token_discount_alpha)))

    print("Whole context, using one of the output strings:")
    test_3 = priority_candidates[-1][0]
    print(test_3)
    print("Score 3: " + str(score_string_odds(model, tokenizer, test_3, token_discount_alpha)))

    print("Done")


def score_string_odds(model, tokenizer, in_string, token_discount_alpha):
    # For debugging; trying to understand terribly poor results I'm getting...
    with ((torch.no_grad())):
        combo1_encode = tokenizer(in_string, return_tensors='pt')

        string_ids = torch.squeeze(combo1_encode['input_ids'])[1:-1]

        # Logits only.
        output = torch.squeeze(model(**combo1_encode).logits)
        # ignore start/end tokens
        output = output[1:-1]

        logsum_term = torch.sum(torch.logsumexp(output, dim=1))

        matched_tokens = output[range(len(string_ids)), string_ids]

        score = torch.sum(matched_tokens) - logsum_term
        # Get back into probability world. Although this is unnecessary for ranking relative candidates.
        score = torch.exp(score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha)))

    return score.item()

if __name__ == "__main__":
    parse_all()
    # build_templates()
    # bert_syntaxify_test()
    # determine_bert_tokens()