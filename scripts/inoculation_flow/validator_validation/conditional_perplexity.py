import argparse
import os, sys
import numpy as np
import pandas as pd
import time
import gc
import re
import random
import copy
import math

import transformers
import torch


KEYPATH = ""
with open(KEYPATH, 'r') as f:
    API_KEY = f.readline()
    API_KEY = API_KEY.rstrip('\n')

# Functional local debug only
TEST_MODEL = "TinyLlama-1.1B-Chat-v1.0"

MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEVICE_STR = 'cpu'

GEN_TEMP = 1.0

ALPHA = 0.8
# Will try using logprobs to calculate some threshold score
# Inspiration: https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/6

# Related: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores

# May try using the forced_decoder_ids option during generation - see what happens.


def manual_generate(pipeline, start_ids, num_new_tokens):
    # Idea on forced generation:
    # https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
    # To help get the logits I want!

    # decoder_ids = []
    # decoder_input_ids = [pipeline.model.config.decoder_start_token_id]

    in_ids = copy.deepcopy(start_ids)

    predicted_ids = []
    # Code adapted from here: https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
    for a in range(num_new_tokens):

        outputs = pipeline.model(input_ids=in_ids)
        logits = outputs.logits[:, -1, :]
        # perform argmax on the last dimension (i.e. greedy decoding)
        predicted_id = logits.argmax(-1)
        predicted_ids.append(predicted_id.item())
        print(pipeline.tokenizer.decode([predicted_id.squeeze()]))
        # Accumulate input ids
        # JL - code was out of date. Only input_ids are used.
        in_ids = torch.concat([in_ids, torch.unsqueeze(predicted_id, dim=0)], dim=1)
        # in_ids = torch.concat([in_ids, predicted_id])


    print(pipeline.tokenizer.decode(predicted_ids))

# Manual generation works. adapt to get logprobs.


def get_prompt_gen_logprob(pipeline, prompt, q1, q2):
    # q1 is substituted into the prompt.
    model_query = [{"role": "user", "content": prompt.format(q1=q1)}]

    # Get the tokens for our desired response.
    test_response = pipeline.tokenizer(q2, padding=False)  # Keeping padding off for now.

    # build forced list.
    # Ignore the first token in this case
    forced_list = []
    for idx in range(1, len(test_response['input_ids'])):
        forced_list.append([idx-1, test_response['input_ids'][idx]])

    test_num_tokens = len(test_response['input_ids']) - 1

    # one-time query
    prompt = pipeline.tokenizer.apply_chat_template(model_query, tokenize=False, add_generation_prompt=True)


    encoded_request = pipeline.tokenizer(prompt, return_tensors="pt",
                                                padding=False)  # Keeping padding off for now.

    # manual_generate(pipeline, encoded_request['input_ids'], 20)

    print("Model Prompt~~~~~~~~~~~~~~")
    print(prompt)

    logits_sequence = []

    # No gradients!
    with torch.no_grad():

        in_ids = copy.deepcopy(encoded_request['input_ids'])

        predicted_ids = []
        # Code adapted from here: https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
        for a in range(test_num_tokens):
            outputs = pipeline.model(input_ids=in_ids)
            logits = outputs.logits[:, -1, :]
            logits_sequence.append(logits)

            # Force the next token to be what I want.
            in_ids = torch.concat([in_ids, torch.tensor([[test_response['input_ids'][a+1]]])], dim=1)
            # in_ids = torch.concat([in_ids, predicted_id])

        del encoded_request
        del in_ids

        # Compute logsumexp.
        logits_sequence = torch.concat(logits_sequence, dim=0)

        # Re-index by our given sequence.

        # Remembering how to use logsumexp:
        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        return get_sequence_odds(logits_sequence, test_response['input_ids'][1:], alpha=ALPHA)


def get_sequence_odds(logits_tensor, selected_tokens_list, alpha=ALPHA):
    # Compare to softmaxing & mult.

    forced_logits = logits_tensor[list(range(len(selected_tokens_list))), selected_tokens_list]

    # Softmax, but in logits world.
    forced_token_logits = forced_logits - torch.logsumexp(logits_tensor, dim=1)

    sequence_prob = torch.exp(torch.sum(forced_token_logits)) / math.pow(float(len(selected_tokens_list)), alpha)

    # # test the traditional way
    # sequence_softmax = torch.softmax(logits_tensor, dim=1)
    # odds_list = sequence_softmax[list(range(len(selected_tokens_list))), selected_tokens_list]
    #
    # sequence_prob2 = torch.prod(odds_list)
    #
    # print("Done")
    return sequence_prob

def scratch():
    pipeline = None
    print("Initializing model & pipeline...")
    # Model, tokenizer, pipeline init
    device_map_setting = DEVICE_STR  # 'auto'  # 'cuda'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        token=API_KEY,
        device_map=device_map_setting
    )

    # Set to eval mode!
    # https://discuss.huggingface.co/t/inference-without-gradient-computation/14449
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO, token=API_KEY)

    if torch.cuda.is_available():
        print("GPU available...")
    else:
        print("No GPU available...")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,  # Probably won't work on cpu...
        tokenizer=tokenizer,
        device_map=device_map_setting,
    )

    test_prompt_str = ("Repeat the provided sentence exactly, word for word."
                       "\nSentence: '{q1}'\n")
    q1 = "Hello world!"
    q2 = "Greetings Earth!"
    q3 = "Goodbye world!"
    q4 = "asdfasl;hirukyau"

    odds1 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q1)
    odds2 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q2)
    odds3 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q3)
    odds4 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q4)

    # Try the inverted
    test_prompt_str = ("Say a sentence that means the opposite of the provided sentence."
                       "\nSentence: '{q1}'\n")
    q1 = "Hello world!"
    q2 = "Greetings Earth!"
    q3 = "Goodbye world!"
    q4 = "asdfasl;hirukyau"

    invert_odds1 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q1)
    invert_odds2 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q2)
    invert_odds3 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q3)
    invert_odds4 = get_prompt_gen_logprob(pipeline, test_prompt_str, q1, q4)

    # Well, the code works. But tinyllama is a crap model, so inverting doesn't get expected result!

    print("Done")




if __name__ == "__main__":
    scratch()
    #
    # test_tensor = torch.tensor([[-1, 1, -1],[1, -1, 2],[3, 1,-5]], dtype=torch.float32)
    # test_tensor = test_tensor * 0.000001
    # selected_tokens = [0, 0, 0]
    #
    # get_sequence_odds(test_tensor, selected_tokens, alpha=1.0)
