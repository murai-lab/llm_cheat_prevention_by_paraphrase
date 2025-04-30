# Jeremy Lim
# Simple wrapper class for masked language modeling

import string
import sys, os

import transformers
import torch
import numpy as np


from transformers import pipeline



# BERT on Huggingface: https://huggingface.co/google-bert/bert-base-uncased
# On BERT special tokens: https://datascience.stackexchange.com/questions/51522/what-is-the-use-of-sep-in-paper-bert
# Distilbert example: https://huggingface.co/distilbert/distilbert-base-uncased
# Only one token at a time! https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/pipelines#transformers.FillMaskPipeline
# https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/bert#transformers.BertForMaskedLM

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
from transformers import BertTokenizer, BertModel

class BERTMLMWrapper:

    def __init__(self):
        # Using distilbert for initial tests
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

        # Try regular bert
        # self.tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = DistilBertForMaskedLM.from_pretrained("bert-base-uncased")

        # Use example:
        # text = "Replace [MASK] by any [MASK] you'd like."
        # encoded_input = self.tokenizer(text, return_tensors='pt')
        # output = self.model(**encoded_input)
        #
        # decode_test = self.tokenizer.decode(output[0])


        print("break")


    def get_prefix_odds(self, prefix_str, eval_str):
        with torch.no_grad():
            # Using a prefix string, turn it into a sequence of mask tokens. Get a sort of "odds score" from the model for this prefix str.
            prefix_encode = self.tokenizer(prefix_str, return_tensors='pt')

            # Bert seems to have a start/end token for each encoding. Take that into account.
            num_masks = prefix_encode.data['input_ids'].shape[1] - 2
            strip_prefix_encode_ids = torch.squeeze(prefix_encode.data['input_ids'][:,1:-1])

            # eval_encode = self.tokenizer(eval_str, return_tensors='pt')

            masks = "[MASK]" * num_masks
            combined_encode = self.tokenizer(masks + eval_str, return_tensors='pt')


            output = self.model(**combined_encode)

            probs = torch.squeeze(torch.softmax(output.logits, dim=2))

            # get probs of the specific prefix tokens.
            # TODO: Reimplement using torch ops.
            probMul = 1.0
            for idx, prefix_id in enumerate(strip_prefix_encode_ids):
                # NOTE: +1 offset. Ignore start/end tokens!
                probMul = probMul * probs[idx + 1, prefix_id]

        return probMul.item()


    # The following makes heavy use of python's format method: https://docs.python.org/3/tutorial/inputoutput.html
    # https://docs.python.org/3/library/string.html#formatspec

    # Should handle most formatting. Not specifically tested though...
    def get_template_score_set(self, format_template_list, format_args=None, format_kwargs=None, token_num_discount=0.8):

        assert format_args is not None or format_kwargs is not None, "Either args or kwargs is required!"
        formatter = string.Formatter()

        scorelist = []
        for fmt_str in format_template_list:
            # Build the template-only string. Remove open/closed curly braces.
            parselist = formatter.parse(fmt_str)

            # use this later. No-nonsense encoding
            with torch.no_grad():
                # Using some syntactic sugar below, inspired by: https://stackoverflow.com/questions/23484091/pass-kwargs-if-not-none
                default_candidate_encode = self.tokenizer(fmt_str.format(*(format_args or ()), **(format_kwargs or {})), return_tensors='pt').data['input_ids']
                # Ignore start/end tokens.
                default_candidate_encode = torch.squeeze(default_candidate_encode)[1:-1]

            masked_string = ""
            positional_count = 0
            mask_token_count = 0
            for p in parselist:
                if len(p[0]) > 0:

                    with torch.no_grad():
                        tokens_encoded = self.tokenizer(p[0], return_tensors='pt')
                        num_masks = tokens_encoded.data['input_ids'].shape[1] - 2

                    masks = "[MASK]" * num_masks
                    masked_string += masks
                    mask_token_count += num_masks

                    # Start search from new position.
                    # start_idx = start_idx + len(p[0])

                # Following python docs; check if we actually have some format substitution point.
                if (p[1] is not None) or (p[2] is not None) or (p[3] is not None):
                    # build mini format string
                    mini_format = "{" # + p[1] + "!" + p[2] + ":" + p[3] + "}"
                    if p[2] is not None:
                        if p[2] != '':
                            mini_format += "!" + p[2]
                    if p[3] is not None:
                        if p[3] != '':
                            mini_format += ":" + p[2]
                    mini_format += '}'
                    # Use p[1] to figure out proper substitution.
                    if p[1] != '':
                        # see if it converts to int.
                        try:
                            arg_pos = int(p[1])
                            subformat = mini_format.format(format_args[arg_pos])
                        except ValueError:
                            # not possible. Assume keyword value.

                            subformat = mini_format.format(format_kwargs[p[1]])
                    else:
                        subformat = mini_format.format(format_args[positional_count])
                        positional_count += 1

                    masked_string += subformat

            print(masked_string)
            # masked string properly built. Now run bert on this template.

            with torch.no_grad():
                combined_encode = self.tokenizer(masked_string, return_tensors='pt')

                # Ignore start/end tokens.
                combined_encode_ids = torch.squeeze(combined_encode.data['input_ids'])[1:-1]
                # Below outputs logits.
                output = torch.squeeze(self.model(**combined_encode).logits)

                # for every masked string, get the logits of only the desired template token.

                # TODO: Reimplement below using torch ops for efficiency
                sum_score = 0
                for idx, desired_token_id in enumerate(default_candidate_encode):
                    if combined_encode_ids[idx] == self.tokenizer.mask_token_id:
                        # logit for desired token

                        # NOTE: +1 offset. Ignore start/end tokens!
                        sum_score += output[idx+1, desired_token_id]

                        # normalize by log(expsum) of all of the values (Compare to "softmax")
                        sum_score -= torch.log(torch.sum(torch.exp(output[idx+1, :])))

                # Apply adjustment for # of masks.
                sum_score = sum_score / torch.pow(torch.tensor(num_masks), torch.tensor(token_num_discount))
                scorelist.append(sum_score.item())

        # Return the computed scores for the whole set.
        return scorelist


def class_test():
    mask_predictor = BERTMLMWrapper()

    # Basic templates - test choosing between them
    prefixes = ["Who is ",
                "What is ",
                "When is ",
                "Where is "]
    # From an example I recorded.
    eval_str = "harlem blues and jazz band instance of?"

    scorelist = []
    for p in prefixes:
        score = mask_predictor.get_prefix_odds(p, eval_str)
        print("String: " + p + eval_str)
        print("Score: " + str(score))
        scorelist.append(score)

    best_item = np.argmax(np.array(scorelist))
    print("Best score: " + str(scorelist[best_item]))
    print("Best string: " + prefixes[best_item] + eval_str)
    # Best score should be 1.2287928257137537e-06

    print("test general updated method. Should have the same ordering.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    templates = ["Who is {}?",
                "What is {}?",
                "When is {}?",
                "Where is {}?"]

    sub_str = "harlem blues and jazz band instance of"

    score_set = mask_predictor.get_template_score_set(templates, (sub_str,),  token_num_discount=0.8)

    for idx, s in enumerate(score_set):
        print("String: " + templates[idx].format(sub_str))
        print("Score: " + str(s))

    best_idx = np.argmax(np.array(score_set))
    print("Best score: " + str(score_set[best_idx]))
    print("Best string: " + templates[best_idx].format(sub_str))

    print("Done")

def debug_test():
    # test_wrapper = BERTMLMWrapper()
    # Testing two masks.
    # unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
    # Try regular bert bert-base-uncased
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    #
    # text = "Replace [MASK] by any [MASK] you'd like."
    #
    # results = unmasker(text)

    # Useful link: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/bert#transformers.BertForMaskedLM

    # How to access token values in a better way...
    text = "Replace [MASK] by any [MASK] you'd like."

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

    # Useful example, inspiration for code: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/bert#transformers.BertForMaskedLM
    encoded_input = tokenizer(text, return_tensors='pt')

    mask_positions = torch.squeeze(encoded_input.data['input_ids'] == tokenizer.mask_token_id).nonzero()
    print("Break")

    with torch.no_grad():
        output = model(**encoded_input)

    # Get predictions for just those mask positions

    print("Break")


if __name__ == "__main__":
    # debug_test()
    class_test()