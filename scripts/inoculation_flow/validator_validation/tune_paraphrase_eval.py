import argparse
import os, sys
import numpy as np
import pandas as pd
import time
import gc
import re
import random


import transformers
import torch

from transformers import LlamaForCausalLM, LlamaTokenizerFast

# Modified for testing the llama models
# using huggingface
# jlim@wpi.edu

KEYPATH = ""
with open(KEYPATH, 'r') as f:
    API_KEY = f.readline()
    API_KEY = API_KEY.rstrip('\n')

# LLaMA model links:
# https://huggingface.co/meta-llama/Llama-2-7b-chat/tree/main
# Nevermind, use this similar model instead:
# https://huggingface.co/meta-llama/Llama-2-7b-hf
# Example code for using transformers with the model. I used this as a starting point:
# https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed

# Try a bigger model...
# TEST_MODEL = "Llama-3.1-8B-Instruct"
# MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"  # Other model does not have a chat template; no chat support?

TEST_MODEL = "Llama-3.2-3B-Instruct"
MODEL_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # Other model does not have a chat template; no chat support?

GEN_TEMP = 1.0

# MAX_NEW_TOKENS = 10  # Should be 10, test 280
MAX_NEW_TOKENS = 1000 # Longer, to allow COT reasoning
EOT_STR = "<|eot_id|>"

DEVICE_STR = 'cuda'

PARSE_RETRIES = 5

# Training sets
PAWS_WIKI_PATH = '/home/jlim/MS_Project/validator_validation/paws_wiki_labeled_final/final/train.tsv'
PAWS_QQP_PATH = '/home/jlim/MS_Project/validator_validation/paws_qqp/output/train.tsv'

# PAWS_WIKI_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_wiki_labeled_final/final/train.tsv'
# PAWS_QQP_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_qqp/output/train.tsv'




def eval_q(pipeline, q1, q2):
    # return the label that the model answers for a given prompt.
    # True means paraphrase, False means no paraphrase.


    # verify_prompt = ("Do the following two questions have the same meaning? "
    #                  "You must answer must start with either \"Yes\" or \"No\"."
    #                  "\nQuestion 1: '{q1}'"
    #                  "\nQuestion 2: '{q2}'\n")

    # Alternate... harder or easier?
    # verify_prompt = ("Do the following two questions have the same answer? "
    #                  "Your answer must begin with either \"Yes\" or \"No\"."
    #                  "\nQuestion 1: {q1}"
    #                  "\nQuestion 2: {q1}\n")

    # more generic prompt... Would this work?
    # verify_prompt = ("Do the following two sentences have the same meaning? "
    #                  "You must answer must start with either \"Yes\" or \"No\"."
    #                  "\nSentence 1: '{q1}'"
    #                  "\nSentence 2: '{q2}'\n")
    
    # Improved COT prompt. Should allow the 3B model to actually work.
    # Not exactly "excellent" at evaluating though.
    verify_prompt = ("Do the following two sentences in quotes have the same meaning? Please explain your answer step by step, and "
                     "provide either \"yes.\" or \"no.\" at the end of your explanation.."
                    "\nSentence 1: '{q1}'"
                    "\nSentence 2: '{q2}'\n")

    verify_msgs = [{"role": "user", "content": verify_prompt.format(q1=q1, q2=q2)}]

    successful_parse = False
    fail_count = 0
    while not successful_parse:

        prompt = pipeline.tokenizer.apply_chat_template(verify_msgs, tokenize=False, add_generation_prompt=True)

        # print("Model Prompt~~~~~~~~~~~~~~")
        # print(prompt)

        encoded_verify_request = pipeline.tokenizer(prompt, return_tensors="pt",
                                                    padding=False)  # Keeping padding off for now.

        # Put on proper device?
        # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
        encoded_verify_request.to(DEVICE_STR)

        generated_output = pipeline.model.generate(**encoded_verify_request, do_sample=True, temperature=GEN_TEMP,
                                                   max_new_tokens=MAX_NEW_TOKENS,
                                                   return_dict_in_generate=True, output_scores=True, output_logits=True,
                                                   pad_token_id=pipeline.tokenizer.eos_token_id)

        # generated_output = generated_output.sequences[0][-MAX_NEW_TOKENS:]
        generated_output = generated_output.sequences[0][encoded_verify_request['input_ids'].shape[1]:]

        

        # Detokenize, convert to lowercase.
        generated_output = pipeline.tokenizer.decode(generated_output).lower()

        # Keep tensor memory clear!
        del encoded_verify_request

        # print("Model Response~~~~~~~~~~~~~~")
        # print(generated_output)

        # if re.search("^yes", generated_output):  # Begin is "^yes"
        #     # yes
        #     print("Parsed yes")
        #     return True
        # elif re.search("^no", generated_output):  # Begin is "^no"
        #     # no
        #     print("Parsed no")
        #     return False

        if re.search(re.escape("yes.<|eot_id|>") + "$", generated_output):  # Begin is "^yes"
            # yes
            # print("Parsed yes")
            return True
        elif re.search(re.escape("no.<|eot_id|>") + "$", generated_output):  # Begin is "^no"
            # no
            # print("Parsed no")
            return False

        else:
            fail_count += 1
            print("Response validation failure. Fail count: " + str(fail_count))
            # print("Failing Validation response:")
            # print(generated_output)
            if fail_count > PARSE_RETRIES:
                print("Could not generate a valid response!")
                return None

# Dummy function for local testing.
def eval_q_debug(pipeline, q1, q2):
    return random.choice([True, False])

def main():
    # Load wiki paraphrase data.
    print("Loading paraphrase data")
    trainwiki_df = pd.read_csv(PAWS_WIKI_PATH, sep='\t')
    trainqqp_df = pd.read_csv(PAWS_QQP_PATH, sep='\t')

    pipeline = None
    print("Initializing model & pipeline...")
    # Model, tokenizer, pipeline init
    device_map_setting = DEVICE_STR  # 'auto'  # 'cuda'
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        token=API_KEY,
        device_map=device_map_setting
    )
    
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

    # need to evaluate both.
    print("Evaluating PAWS Wiki")
    wiki_num_answered = 0
    wiki_tp = 0
    wiki_tn = 0
    wiki_fn = 0
    wiki_fp = 0
    for i in range(trainwiki_df.shape[0]):
        if i % 100 == 0:
            print("Idx: " + str(i))
            sys.stdout.flush()
        ground_truth = trainwiki_df.iloc[i, 3]
        result = eval_q(pipeline, trainwiki_df.iloc[i, 1], trainwiki_df.iloc[i, 2])
        if result is not None:
            wiki_num_answered += 1
            # Accumulate confusion matrix.
            if ground_truth == 1:
                if result is True:
                    wiki_tp += 1
                else:
                    wiki_fn += 1
            else:
                if result is True:
                    wiki_fp += 1
                else:
                    wiki_tn += 1

    print("Eval results for PAWS Wiki: ")
    print("Num answered/num pairs: " + str(wiki_num_answered) + "/" + str(trainwiki_df.shape[0]))
    print("Accuracy: " + str((wiki_tp + wiki_tn)/wiki_num_answered ))
    print("Sensitivity: " + str((wiki_tp)/(wiki_tp + wiki_fn)))
    print("Specificity: " + str((wiki_tn)/(wiki_tn + wiki_fp)))

    # TODO: Make into function instead of copypasta.
    # This dataset is closer to my desired domain.
    print("Evaluating PAWS qqp")
    qqp_num_answered = 0
    qqp_tp = 0
    qqp_tn = 0
    qqp_fn = 0
    qqp_fp = 0
    for j in range(trainqqp_df.shape[0]):
        if j % 100 == 0:
            print("Idx: " + str(j))
            sys.stdout.flush()
        ground_truth = trainqqp_df.iloc[j, 3]
        result = eval_q(pipeline, trainqqp_df.iloc[j, 1], trainqqp_df.iloc[j, 2])
        if result is not None:
            qqp_num_answered += 1
            # Accumulate confusion matrix.
            if ground_truth == 1:
                if result is True:
                    qqp_tp += 1
                else:
                    qqp_fn += 1
            else:
                if result is True:
                    qqp_fp += 1
                else:
                    qqp_tn += 1

    print("Eval results for PAWS qqp: ")
    print("Num answered/num pairs: " + str(qqp_num_answered) + "/" + str(trainqqp_df.shape[0]))
    print("Accuracy: " + str((qqp_tp + qqp_tn)/qqp_num_answered ))
    print("Sensitivity: " + str((qqp_tp)/(qqp_tp + qqp_fn)))
    print("Specificity: " + str((qqp_tn)/(qqp_tn + qqp_fp)))

    # Roughly 60,000 pairs to evaluate...
    # Bootstrapping/batching required to make this fast?

if __name__ == "__main__":
    main()