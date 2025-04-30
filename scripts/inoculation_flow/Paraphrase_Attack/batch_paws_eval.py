# Test evaluation prompt against full PAWS.
import argparse
import sys, os

import numpy as np
import pandas as pd
import time
import gc
import re
import random
import copy
import math
from matplotlib import pyplot as plt
import pickle
import uuid


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
PRINT_INTERVAL = 100

EOT_STR = "<|eot_id|>"

DEVICE_STR = 'cuda'

# Training sets
PAWS_WIKI_PATH = '/home/jlim/MS_Project/validator_validation/paws_wiki_labeled_final/final/train.tsv'
PAWS_QQP_PATH = '/home/jlim/MS_Project/validator_validation/paws_qqp/output/train.tsv'

MAX_NEW_TOKENS = 10  # for short answer

def build_batches(args):

    batchcount = args.batches
    wiki_dir = args.paws_wiki_dir
    qqp_dir = args.paws_qqp_dir
    output_dir = args.save_dir
    subsample_count = args.subsample_count  # If none, do the whole set.

    # make this output directory.
    os.makedirs(output_dir, exist_ok=True)

    # Read files

    print("Loading paraphrase data")
    trainwiki_df = pd.read_csv(wiki_dir, sep='\t')
    trainqqp_df = pd.read_csv(qqp_dir, sep='\t')

    # Deduplicate!
    print("Deduplicating wiki...")
    print("wiki shape: " + str(trainwiki_df.shape))
    count_drop = 0
    for i in range(trainwiki_df.shape[0]-1, -1, -1):
        if(trainwiki_df.iloc[i, 1] == trainwiki_df.iloc[i, 2]):
            trainwiki_df.drop(trainwiki_df.index[i], inplace=True)
            count_drop += 1

    print("Records removed from wiki: " + str(count_drop))
    print("wiki shape: " + str(trainwiki_df.shape))

    print("Deduplicating qqp...")
    print("qqp shape: " + str(trainqqp_df.shape))
    count_drop = 0
    for i in range(trainqqp_df.shape[0]-1, -1, -1):
        if(trainqqp_df.iloc[i, 1] == trainqqp_df.iloc[i, 2]):
            trainqqp_df.drop(trainqqp_df.index[i], inplace=True)
            count_drop += 1

    print("Records removed from qqp: " + str(count_drop))
    print("qqp shape: " + str(trainqqp_df.shape))

    # If we subsample, take a balanced subsample of subsample_count from each dataset.
    if subsample_count is not None:
        # Force batchcount = 1
        batchcount = 1

        true_p_wiki = trainwiki_df[trainwiki_df['label'] == 1]
        true_p_qqp = trainqqp_df[trainqqp_df['label'] == 1]

        print("Num positive wiki: " + str(true_p_wiki.shape[0]))
        print("Num positive qqp: " + str(true_p_qqp.shape[0]))


        false_p_wiki = trainwiki_df[trainwiki_df['label'] == 0]
        false_p_qqp = trainqqp_df[trainqqp_df['label'] == 0]

        print("Num negative wiki: " + str(false_p_wiki.shape[0]))
        print("Num negative qqp: " + str(false_p_qqp.shape[0]))

        # Subsample and build balanced mini datasets randomly.
        wiki_subdata = pd.concat([true_p_wiki.sample(n=int(subsample_count/2)), false_p_wiki.sample(n=int(subsample_count/2))])
        qqp_subdata = pd.concat([true_p_qqp.sample(n=int(subsample_count/2)), false_p_qqp.sample(n=int(subsample_count/2))])

        # Desired data structure:
        # uuid
        # text1
        # text2
        # gt_is_paraphrase
        # eval_is_paraphrase
    else:
        wiki_subdata = trainwiki_df
        qqp_subdata = trainqqp_df

    comparison_listing = []

    # Wikidata
    for i in range(wiki_subdata.shape[0]):
        ground_truth = wiki_subdata.iloc[i, 3]
        comparison_listing.append({
            'uuid': str(uuid.uuid4()),
            'source': 'wiki',
            'text1': wiki_subdata.iloc[i, 1],
            'text2': wiki_subdata.iloc[i, 2],
            'ground_truth_is_paraphrase': ground_truth,
            'eval_is_paraphrase': None,  # Will be computed by our evaluation step.
        })

    # qqp
    for i in range(qqp_subdata.shape[0]):
        ground_truth = qqp_subdata.iloc[i, 3]
        comparison_listing.append({
            'uuid': str(uuid.uuid4()),
            'source': 'qqp',
            'text1': qqp_subdata.iloc[i, 1],
            'text2': qqp_subdata.iloc[i, 2],
            'ground_truth_is_paraphrase': bool(ground_truth),
            'eval_is_paraphrase': None,  # Will be computed by our evaluation step.
        })

    # split up into the proper number of batches.
    batch_sizes = int(len(comparison_listing) / batchcount)
    for j in range(batchcount):
        
        if j == batchcount - 1:
            # Grab remainder.
            segment = comparison_listing[j*batch_sizes:]
        else:
            segment = comparison_listing[j*batch_sizes:(j+1)*batch_sizes]

        savepath = os.path.join(output_dir, "batch_" + str(j) + '.pickle')
        print("Saving " + str(len(segment)) + " records to " + savepath) 
        with open(os.path.join(output_dir, "batch_" + str(j) + '.pickle'), 'wb') as f:
            pickle.dump(segment, f)


# Handle actually evaulating a specific paraphrasing technique
def eval_q(pipeline, q1, q2):
    # return the label that the model answers for a given prompt.
    # True means paraphrase, False means no paraphrase.

    # Fixing: Used the wrong prompt!
    verify_prompt = ("Do sentence 1 and sentence 2 have extremely different meanings? "
        "Remember that some sentences that are phrased differently have the same meaning. "
        "The sentences may appear different, but the wording may actually imply a very similar meaning. "
        "For your answer, please state either \"yes.\" or \"no.\" at the start of your answer."
        "\nSentence 1: '{q1}'"
        "\nSentence 2: '{q2}'\n")
    

    verify_msgs = [{"role": "user", "content": verify_prompt.format(q1=q1, q2=q2)}]

    prompt = pipeline.tokenizer.apply_chat_template(verify_msgs, tokenize=False, add_generation_prompt=True)

    # print("Model Prompt~~~~~~~~~~~~~~")
    # print(prompt)

    encoded_verify_request = pipeline.tokenizer(prompt, return_tensors="pt",
                                                padding=False)  # Keeping padding off for now.

    # No gradients!
    with torch.no_grad():
        # Put on proper device?
        # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
        encoded_verify_request.to(DEVICE_STR)

        generated_output = pipeline.model.generate(**encoded_verify_request, do_sample=False, temperature=GEN_TEMP,
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

    if re.search("^yes", generated_output):  # Begin is "^yes"
        # yes
        # print("Parsed yes")
        # return True
        return False
    elif re.search("^no", generated_output):  # Begin is "^no"
        # no
        # print("Parsed no")
        # return False
        return True
    else:
        print("Could not generate a valid response!")
        return None


def run_batch(args, api_key):

    batchpath = args.batch_file
    print("Running batch from path: " + batchpath)

    with open(batchpath, 'rb') as f:
        batch_set = pickle.load(f)

    print("Processing batch from file: " + batchpath)

    print("Initializing...")
    # Set up model & dependencies
    device_map_setting = DEVICE_STR  # 'auto'  # 'cuda'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        token=api_key,
        device_map=device_map_setting
    )

    # Set to eval mode!
    # https://discuss.huggingface.co/t/inference-without-gradient-computation/14449
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO, token=api_key)

    if torch.cuda.is_available():
        print("GPU available...")
    else:
        print("No GPU available...")
        print("You're probably not running in the right context. Exiting...")
        exit(1)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,  # Probably won't work on cpu...
        tokenizer=tokenizer,
        device_map=device_map_setting,
    )

    for idx, comparison in enumerate(batch_set):
        if (idx + 1) % PRINT_INTERVAL == 0:
            print("Evaluation: " + str(idx + 1) + "/" + str(len(batch_set)))
        # Evaluate
        # record answer
        eval_result = eval_q(pipeline, comparison['text1'], comparison['text2'])
        # it may be none; meaning a noncompliant answer.
        comparison['eval_is_paraphrase'] = eval_result

    with open(args.batch_result_path, 'wb') as f:
        pickle.dump(batch_set, f)  # This is modified by this process.

def result_stats(results_file):

    with open(results_file, 'rb') as f:
        result_data = pickle.load(f)

    # Collect confusion matrix values.
    wiki_count = 0
    wiki_num_answered = 0
    wiki_tp = 0
    wiki_tn = 0
    wiki_fn = 0
    wiki_fp = 0

    wiki_num_para = 0

    qqp_count = 0
    qqp_num_answered = 0
    qqp_tp = 0
    qqp_tn = 0
    qqp_fn = 0
    qqp_fp = 0

    qqp_num_para = 0

    for comparison in result_data:
        result = comparison['eval_is_paraphrase']
        ground_truth = comparison['ground_truth_is_paraphrase']
        if comparison['source'] == 'wiki':
            wiki_count += 1
            if result is not None:
                wiki_num_answered += 1
                # Accumulate confusion matrix.
                if ground_truth == 1:
                    wiki_num_para += 1
                    if result is True:
                        wiki_tp += 1
                    else:
                        wiki_fn += 1
                else:
                    if result is True:
                        wiki_fp += 1
                    else:
                        wiki_tn += 1
        else: # qqp
            qqp_count += 1
            if result is not None:
                qqp_num_answered += 1
                # Accumulate confusion matrix.
                if ground_truth == 1:
                    qqp_num_para += 1
                    if result is True:
                        qqp_tp += 1
                    else:
                        qqp_fn += 1
                else:
                    if result is True:
                        qqp_fp += 1
                    else:
                        qqp_tn += 1


    print("Eval results for PAWS Wiki: ")

    print("Num answered/num pairs: " + str(wiki_num_answered) + "/" + str(wiki_count))
    print("Paraphrase proportion: " + str(wiki_num_para/wiki_count))
    
    print("TP: " + str(wiki_tp) + "; FP: " + str(wiki_fp) + "; TN: " + str(wiki_tn) + "; FN: " + str(wiki_fn))
    
    print("Accuracy: " + str((wiki_tp + wiki_tn)/wiki_num_answered ))
    try:
        print("Sensitivity: " + str((wiki_tp)/(wiki_tp + wiki_fn)))
    except ZeroDivisionError:
        pass

    try:
        print("Specificity: " + str((wiki_tn)/(wiki_tn + wiki_fp)))
    except ZeroDivisionError:
        pass
    try:
        print("True Positive Rate: " + str((wiki_tp)/(wiki_tp + wiki_fp)))
    except ZeroDivisionError:
        pass


    print("Eval results for PAWS qqp: ")
    print("Num answered/num pairs: " + str(qqp_num_answered) + "/" + str(qqp_count))
    print("Paraphrase proportion: " + str(qqp_num_para/qqp_count))
    print("TP: " + str(qqp_tp) + "; FP: " + str(qqp_fp) + "; TN: " + str(qqp_tn) + "; FN: " + str(qqp_fn))
    print("Accuracy: " + str((qqp_tp + qqp_tn)/qqp_num_answered ))
    try:
        print("Sensitivity: " + str((qqp_tp)/(qqp_tp + qqp_fn)))
    except ZeroDivisionError:
        pass

    try:
        print("Specificity: " + str((qqp_tn)/(qqp_tn + qqp_fp)))
    except ZeroDivisionError:
        pass

    try:
        print("True Positive Rate: " + str((qqp_tp)/(qqp_tp + qqp_fp)))
    except ZeroDivisionError:
        pass


def main():
    print("Start")
    parser = argparse.ArgumentParser()
    # Build batches
    parser.add_argument("--batches", "-b", type=int)
    parser.add_argument("--subsample_count", "-n", type=int, default=None)
    parser.add_argument("--paws_wiki_dir", "-w", type=str)
    parser.add_argument("--paws_qqp_dir", "-q", type=str)
    parser.add_argument("--save_dir", "-s", type=str)

    # Process batch
    parser.add_argument("--batch_file", "-f", type=str)
    parser.add_argument("--batch_result_path", "-r", type=str)

    # See results
    parser.add_argument("--result_stats_path", "-p", type=str)

    print("Parsing args...")
    args = parser.parse_args()

    if args.result_stats_path:
        result_stats(args.result_stats_path)
    elif args.batch_file and args.batch_result_path:  # Process single batch file.
        print("Running batch")

        # JL - moving imports here as login node doesn't cooperate with these imports...
        print("Preparing imports...")
        # A little nasty, but was having issues before when running on login node, and it hung on these imports...
        # https://stackoverflow.com/questions/11990556/how-to-make-global-imports-from-a-function
        global transformers
        global torch
        global LlamaForCausalLM
        global LlamaTokenizerFast

        import transformers
        import torch
        from transformers import LlamaForCausalLM, LlamaTokenizerFast

        KEYPATH = "/home/jlim/MS_Project/MMLU_Baseline/Huggingface_free"
        with open(KEYPATH, 'r') as f:
            API_KEY = f.readline()
            API_KEY = API_KEY.rstrip('\n')

        run_batch(args, api_key=API_KEY)

    else:  # assume we're making a batch.
        if args.batches and args.paws_wiki_dir and args.paws_qqp_dir and args.save_dir:
            print("Building batches")
            build_batches(args)
        else:
            print("Missing arguments!")

if __name__ == "__main__":
    main()

# Quick eval notes:
# Ok, test stats make more sense now; correct prompt
# Though the TPR is not excellent:
# Eval results for PAWS Wiki: 
# Num answered/num pairs: 400/400
# TP: 144; FP: 93; TN: 107; FN: 56
# Accuracy: 0.6275
# Sensitivity: 0.72
# Specificity: 0.535
# True Positive Rate: 0.6075949367088608
# Eval results for PAWS qqp: 
# Num answered/num pairs: 400/400
# TP: 102; FP: 64; TN: 136; FN: 98
# Accuracy: 0.595
# Sensitivity: 0.51
# Specificity: 0.68
# True Positive Rate: 0.6144578313253012

