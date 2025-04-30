# Jeremy Lim
# Perform beam search on BERT family models for seed question generation
import copy
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


def determine_bert_tokens():
    # use the brown corpus along with POS tagging, determine words to restrict bert to based on the tags
    brown_path = "/home/jeremy/nltk_data/corpora/brown"

    # Wikipedia actually has a decent list of the tags used: https://en.wikipedia.org/wiki/Brown_Corpus
    # Brown manual: http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM

    # A word is included if it had been tagged with one of these at least once
    # this is built from the manual: http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM
    include_tags = set(['at', # Articles
                    'do', 'dod', 'doz', # do, did, does. Special verb
                    'in', # Prepositions.
                    'be', 'bed', 'bedz', 'beg', 'bem', 'ben', 'ber', 'bez', # be variants. Special verb
                    'to', # To
                    'wdt', 'wp$', 'wpo', 'wps', 'wql', 'wrb', # All variants of "wh" words.
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
                    # if components[1] == "wrb":
                    #     print("Break")
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

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # roberta
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # deberta
    # tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

    # Determine the token idxs for each of the vocab words.
    keep_idxs = set()
    word_to_idx = {}
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
            # Only doing single token words for now.
            word_to_idx[word] = idx

    print("Number of tokens kept: " + str(len(keep_idxs)))
    # print("Done")

    return keep_idxs, word_to_idx

def bert_syntaxify_wh_search(example_head="George Washington", example_relation="date of birth", max_mask_num=3, n_beams=10, token_discount_alpha=0.8):

    # JL notes:
    # token_discount_alpha = 0.8

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    # Real bert
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

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
    keep_idxs, word_to_idxs = determine_bert_tokens()
    for keep in keep_idxs:
        mask_arr[keep] = 1.0

    print("Vocab mask built.")

    candidate_seeds = []

    min_mask_num = 0
    # build combos
    eval_count = 0

    # "Wh" words.
    wh_words = ["who", "what", "when", "where", "why"]
    wh_words = [word_to_idxs[x] for x in wh_words]

    wh_mask_arr = np.zeros(num_output_logits)
    for keep in wh_words:
        wh_mask_arr[keep] = 1.0


    do_random = True

    # JL - 10/21/24 - create a mask that adds head/relation tokens to it.
    denom_keep_list = list(keep_idxs)
    denom_keep_list = denom_keep_list + head_ids + relation_ids + punctuation_ids
    # denom_mask = np.copy(mask_arr)
    # for keep in head_ids:
    #     denom_mask[keep] = 1.0
    # for keep in relation_ids:
    #     denom_mask[keep] = 1.0
    # for keep in punctuation_ids:
    #     denom_mask[keep] = 1.0

    # 0-3 pre
    # 0-3 middle
    # 0-3 post

    for pre_masks in range(0, max_mask_num+1):
        for middle_masks in range(min_mask_num, max_mask_num+1):
            for post_masks in range(min_mask_num, max_mask_num+1):
                print(str(eval_count) + "/" + str(2*pow(max_mask_num+1, 2)*(max_mask_num+1)))
                # eval {masks} h {masks} r {masks}?

                if(eval_count == 8):
                    print("Break")

                # n_beams = 5
                candidate_tokens = []

                # FORWARD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                combo1 = ("[MASK] " * pre_masks) + example_head + (" [MASK]" * middle_masks) + " " + example_relation + (" [MASK]" * post_masks) + "?"
                print(combo1)

                with ((torch.no_grad())):

                    combo1_encode = tokenizer(combo1, return_tensors='pt')
                    start_encode = combo1_encode['input_ids']

                    candidate_tokens = [(start_encode, torch.tensor(-100000000000))]  # all masks initially.

                    # go left to right
                    for mask_idx in range(pre_masks+middle_masks+post_masks):
                        # logic to determine where we check the mask. We move L->R
                        if mask_idx > (pre_masks + middle_masks - 1):
                            # post-mask
                            test_idx = mask_idx + head_token_count + relation_token_count
                        elif mask_idx > (pre_masks - 1):
                            # Middle-mask
                            test_idx = mask_idx + head_token_count
                        else:  # pre-mask
                            test_idx = mask_idx

                        sub_candidate_set = []

                        for beam in candidate_tokens:
                            # evaluate candidate
                            output = torch.squeeze(model(input_ids=beam[0], attention_mask=combo1_encode['attention_mask']).logits)
                            # ignore start/end tokens
                            output = output[1:-1]

                            beam_tokens = beam[0]
                            beam_tokens_no_terminals = torch.squeeze(beam_tokens)[1:-1]

                            # compute score components, but exclude our current token for now.
                            # logsum_term = torch.sum(torch.logsumexp(output, dim=1))

                            # JL 10/21: Adjust logsum term, only include chosen vocabulary.
                            logsum_term = torch.sum(torch.logsumexp(output[:, denom_keep_list], dim=1))

                            # Calculate some components of the score.
                            before_matched_tokens = None
                            after_matched_tokens = None
                            if mask_idx > 0:
                                before_matched_tokens = output[range(0,mask_idx), beam_tokens_no_terminals[:mask_idx]]

                            if mask_idx < output.shape[0]:
                                after_matched_tokens = output[range(mask_idx+1, output.shape[0]), beam_tokens_no_terminals[mask_idx+1:]]

                            # a single beam can only provide at most n candidates, so we do topk
                            if mask_idx == 0 and False:
                                # Start with "Wh" words first.
                                vals_indices = torch.topk(output[test_idx] * wh_mask_arr, n_beams)
                            else:
                                # vals_indices = torch.topk(output[test_idx] * mask_arr, n_beams)
                                # JL - fixing masking logic. For top-k selection, use mask arr to adjust irrelevant vocabto minimum
                                mask_arr_inv = np.logical_not(mask_arr).astype(np.float64)
                                vals_indices = torch.topk(output[test_idx] * mask_arr + mask_arr_inv * torch.min(output[test_idx]).item(), n_beams)

                            # add the top k candidates to the candidate beams.
                            # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor

                            indices_tensor = torch.squeeze(vals_indices[1])
                            for p_idx in range(n_beams):
                                # Get the score for this candidate.
                                score = output[mask_idx, indices_tensor[p_idx]] - logsum_term
                                if before_matched_tokens is not None:
                                    score += torch.sum(before_matched_tokens)
                                if after_matched_tokens is not None:
                                    score += torch.sum(after_matched_tokens)

                                # sequence length normalization
                                score = score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha))

                                # score = torch.exp(score)
                                #
                                # assert 0.0 <= score <= 1.0

                                sub_candidate = beam_tokens.clone().detach()
                                sub_candidate[0, 1+test_idx] = indices_tensor[p_idx]  # sub in the chosen word.
                                sub_candidate_set.append((sub_candidate, score))

                        # now get top k of the sub-candidate_set
                        sub_candidate_set.sort(key=lambda x: x[1]) # sort by score
                        ordered_sub_candidates = list(reversed(sub_candidate_set))
                        candidate_tokens = ordered_sub_candidates[:n_beams]

                    # n candidates created. Decode them.
                    for c in candidate_tokens:
                        candidate_str = tokenizer.decode(torch.squeeze(c[0])[1:-1])

                        candidate_seeds.append((candidate_str, c[1]))

                # JL notes:
                # Results improving with beam search, L-R. perhaps more randomized search techniques may help?
                # Models not good when many masks are present. Exhaustive search required?

                # # REVERSE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # # eval the reverse: {masks} r {masks} h {masks}?
                combo2 = ("[MASK] " * pre_masks) + example_relation + (" [MASK]" * middle_masks) + " " + example_head + (" [MASK]" * post_masks) + "?"
                print(combo2)
                # TODO: Do better than copypasta!
                with ((torch.no_grad())):

                    combo2_encode = tokenizer(combo2, return_tensors='pt')
                    start_encode = combo2_encode['input_ids']

                    candidate_tokens = [(start_encode, torch.tensor(-100000000000))]  # all masks initially.

                    # go left to right
                    for mask_idx in range(pre_masks+middle_masks+post_masks):
                        # logic to determine where we check the mask. We move L->R
                        if mask_idx > (pre_masks + middle_masks - 1):
                            # post-mask
                            test_idx = mask_idx + head_token_count + relation_token_count
                        elif mask_idx > (pre_masks - 1):
                            # Middle-mask
                            test_idx = mask_idx + relation_token_count
                        else:  # pre-mask
                            test_idx = mask_idx

                        sub_candidate_set = []

                        for beam in candidate_tokens:
                            # evaluate candidate
                            output = torch.squeeze(model(input_ids=beam[0], attention_mask=combo1_encode['attention_mask']).logits)
                            # ignore start/end tokens
                            output = output[1:-1]

                            beam_tokens = beam[0]
                            beam_tokens_no_terminals = torch.squeeze(beam_tokens)[1:-1]

                            # compute score components, but exclude our current token for now.
                            # logsum_term = torch.sum(torch.logsumexp(output, dim=1))

                            # JL 10/21: Adjust logsum term, only include chosen vocabulary.
                            logsum_term = torch.sum(torch.logsumexp(output[:, denom_keep_list], dim=1))

                            # Calculate some components of the score.
                            before_matched_tokens = None
                            after_matched_tokens = None
                            if mask_idx > 0:
                                before_matched_tokens = output[range(0,mask_idx), beam_tokens_no_terminals[:mask_idx]]

                            if mask_idx < output.shape[0]:
                                after_matched_tokens = output[range(mask_idx+1, output.shape[0]), beam_tokens_no_terminals[mask_idx+1:]]

                            # a single beam can only provide at most n candidates, so we do topk
                            if mask_idx == 0 and False:
                                # Start with "Wh" words first.
                                vals_indices = torch.topk(output[test_idx] * wh_mask_arr, n_beams)
                            else:
                                # vals_indices = torch.topk(output[test_idx] * mask_arr, n_beams)
                                # JL - fixing masking logic. For top-k selection, use mask arr to adjust irrelevant vocabto minimum
                                mask_arr_inv = np.logical_not(mask_arr).astype(np.float64)
                                vals_indices = torch.topk(output[test_idx] * mask_arr + mask_arr_inv * torch.min(output[test_idx]).item(), n_beams)

                            # add the top k candidates to the candidate beams.
                            # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor

                            indices_tensor = torch.squeeze(vals_indices[1])
                            for p_idx in range(n_beams):
                                # Get the score for this candidate.
                                score = output[mask_idx, indices_tensor[p_idx]] - logsum_term
                                if before_matched_tokens is not None:
                                    score += torch.sum(before_matched_tokens)
                                if after_matched_tokens is not None:
                                    score += torch.sum(after_matched_tokens)

                                # sequence length normalization
                                score = score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha))

                                # score = torch.exp(score)
                                #
                                # assert 0.0 <= score <= 1.0

                                sub_candidate = beam_tokens.clone().detach()
                                sub_candidate[0, 1+test_idx] = indices_tensor[p_idx]  # sub in the chosen word.
                                sub_candidate_set.append((sub_candidate, score))

                        # now get top k of the sub-candidate_set
                        sub_candidate_set.sort(key=lambda x: x[1]) # sort by score
                        ordered_sub_candidates = list(reversed(sub_candidate_set))
                        candidate_tokens = ordered_sub_candidates[:n_beams]

                    # n candidates created. Decode them.
                    for c in candidate_tokens:
                        candidate_str = tokenizer.decode(torch.squeeze(c[0])[1:-1])

                        candidate_seeds.append((candidate_str, c[1]))

                eval_count += 2

    # re-evaluate
    print("Evaluating fixed form of all candidates...")

    priority_candidates = []
    for c in candidate_seeds:
        score = score_string_odds(model, tokenizer, c[0], token_discount_alpha, denom_keep_list=denom_keep_list)
        priority_candidates.append((c[0], score))

    # priority_candidates = sorted(candidate_seeds, key=lambda x: x[1].item())
    # priority_candidates = sorted(candidate_seeds, key=lambda x: x[1])

    priority_candidates = sorted(priority_candidates, key=lambda x: x[1])

    for p in priority_candidates:
        print(p[0] + " ; " + str(p[1]))

    print("Sanity: Try some human-generated strings")

    # test_1 = "George Washington had what date of birth?"
    #
    # print(test_1)
    # print("Score 1: " + str(score_string_odds(model, tokenizer, test_1, token_discount_alpha, denom_keep_list=denom_keep_list)))

    # UPDATE: Had is not in the vocabulary! it is a general verb.

    # Re-writing another human string:
    test_1 = "George Washington date of birth is when?"

    print(test_1)
    print("Score 1: " + str(
        score_string_odds(model, tokenizer, test_1, token_discount_alpha, denom_keep_list=denom_keep_list)))

    test_2 = "When was the date of birth of George Washington?"
    print(test_2)
    print("Score 2: " + str(score_string_odds(model, tokenizer, test_2, token_discount_alpha, denom_keep_list=denom_keep_list)))

    print("Whole context, using one of the output strings:")
    test_3 = priority_candidates[-1][0]
    print(test_3)
    print("Score 3: " + str(score_string_odds(model, tokenizer, test_3, token_discount_alpha, denom_keep_list=denom_keep_list)))

    # look at the second bes? better/worse?
    print("Whole context, look at the second best:")
    test_3 = priority_candidates[-2][0]
    print(test_3)
    print("Score 3: " + str(score_string_odds(model, tokenizer, test_3, token_discount_alpha, denom_keep_list=denom_keep_list)))

    print("Done")


# def bert_beam_search(model, tokenizer, example_head, example_relation, pre_masks, middle_masks, post_masks):
#     # Search from left to right.
#     n_beams = 5
#
#     # FORWARD~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     combo1 = ("[MASK] " * pre_masks) + example_head + (" [MASK]" * middle_masks) + " " + example_relation + (
#                 " [MASK]" * post_masks) + "?"
#     print(combo1)
#
#     with ((torch.no_grad())):
#         combo1_encode = tokenizer(combo1, return_tensors='pt')
#
#         # Logits only.
#         output = torch.squeeze(model(**combo1_encode).logits)
#         # ignore start/end tokens
#         output = output[1:-1]
#
#         # Greedy: Choose tokens that maximize the "probability"; can try other search methods in the future.
#         pre_scores = torch.max(output[:pre_masks] * mask_arr, dim=1)
#         # Index only the fixed words for head scores
#         head_scores = output[range(pre_masks, pre_masks + len(head_ids)), head_ids]
#         middle_scores = torch.max(
#             output[(pre_masks + head_token_count):(pre_masks + head_token_count) + middle_masks] * mask_arr, dim=1)
#         # Index only the fixed words for relation scores
#         relation_scores = output[range((pre_masks + head_token_count + middle_masks),
#                                        (pre_masks + head_token_count + middle_masks) + len(relation_ids)), relation_ids]
#         end_scores = torch.max(
#             output[(pre_masks + head_token_count + middle_masks + relation_token_count):-1] * mask_arr, dim=1)
#         punctuation_scores = output[-1, punctuation_ids[0]]
#
#         # sum all scores. Then compute logsumexp to normalize.
#         # Note: Logsumexp trick? https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
#
#         # Debugging score issues
#
#         # test_first_score = torch.exp(pre_scores.values[0] -torch.sum(torch.logsumexp(output[:1], dim=1)))
#
#         logsum_term = torch.sum(torch.logsumexp(output, dim=1))
#
#         score = torch.sum(pre_scores.values) + \
#                 torch.sum(head_scores) + \
#                 torch.sum(middle_scores.values) + \
#                 torch.sum(relation_scores) + \
#                 torch.sum(end_scores.values) + \
#                 torch.sum(punctuation_scores) - \
#                 logsum_term
#
#         # Get back into probability world. Although this is unnecessary for ranking relative candidates.
#         score = torch.exp(score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha)))
#
#         # Decode output sentence. Include start/end tokens.
#         token_set = pre_scores.indices.tolist() + head_ids + middle_scores.indices.tolist() + relation_ids + end_scores.indices.tolist() + punctuation_ids
#
#         candidate_str = tokenizer.decode(token_set)
#
#         candidate_seeds.append((candidate_str, score))

def score_string_odds(model, tokenizer, in_string, token_discount_alpha, denom_keep_list=None):
    # For debugging; trying to understand terribly poor results I'm getting...
    with ((torch.no_grad())):
        combo1_encode = tokenizer(in_string, return_tensors='pt')

        string_ids = torch.squeeze(combo1_encode['input_ids'])[1:-1]

        # Logits only.
        output = torch.squeeze(model(**combo1_encode).logits)
        # ignore start/end tokens
        output = output[1:-1]

        if denom_keep_list is not None:
            # sanity check
            # for id in string_ids:
            #     if id not in denom_keep_list:
            #         raise Exception("Id not in keep list!")

            logsum_term = torch.sum(torch.logsumexp(output[:, denom_keep_list], dim=1))
        else:
            logsum_term = torch.sum(torch.logsumexp(output, dim=1))

        matched_tokens = output[range(len(string_ids)), string_ids]

        score = torch.sum(matched_tokens) - logsum_term
        # Get back into probability world. Although this is unnecessary for ranking relative candidates.
        score = torch.exp(score / torch.pow(torch.tensor(output.shape[0]), torch.tensor(token_discount_alpha)))

    return score.item()

if __name__ == "__main__":
    # parse_all()
    # build_templates()
    # bert_syntaxify_wh_search()
    # Standard
    bert_syntaxify_wh_search(example_head="George Washington", example_relation="date of birth", max_mask_num=3,
                             n_beams=10, token_discount_alpha=0.8)
    # Lower alpha
    # bert_syntaxify_wh_search(example_head="George Washington", example_relation="date of birth", max_mask_num=3,
    #                          n_beams=10, token_discount_alpha=0.4)
    # Higher alpha
    # bert_syntaxify_wh_search(example_head="George Washington", example_relation="date of birth", max_mask_num=3,
    #                          n_beams=10, token_discount_alpha=1.6)

    # Alpha = 0.0 (No length correction)
    # bert_syntaxify_wh_search(example_head="George Washington", example_relation="date of birth", max_mask_num=3,
    #                          n_beams=10, token_discount_alpha=0.0)

    # determine_bert_tokens()