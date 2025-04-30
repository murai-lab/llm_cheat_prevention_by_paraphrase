# Test evaluation prompt against full PAWS.
import argparse
import sys, os
import re
import itertools

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
import datetime


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

MAX_PARAPHRASE_TOKENS = 1000  # About twice the length of the longest token.

EVAL_NEW_TOKENS = 10

ANSWER_PREFIX = "The best answer is {}"

ANSWER_REG = "The best answer is [ABCD]"

def build_batches(args):

    # assume this format:
    # param_set = {
    #     'model': OPENAI_TEST_MODEL,
    #     'messages': msgs,
    #     'logprobs': False,  # False for now.
    #     # 'max_tokens': RESPONSE_MAX_NEW_TOKENS, # GPT turbo
    #     'temperature': GEN_TEMP,
    #     'max_completion_tokens': MAX_NEW_TOKENS,  # o3 mini
    #     # 'top_logprobs': 20
    # }

    # q_info = {
    #     'uuid': str(uuid.uuid4()),  # Nearly guaranteed unique.
    #     'subject': subject,
    #     'questio_only': data_df.iloc[q_idx, 0],
    #     'question': format_question_str,
    #     'answer': answer_label,
    #     'model_query': format_question_str,
    #     'param_set': param_set,
    #     'model_response': None
    #     'paras_list'
    #     'num_paras'
    # }
    os.makedirs(args.batch_create_path, exist_ok=True)

    with open(args.response_file, 'rb') as f:
        q_info_list = pickle.load(f)
        
    noncomply_count = 0
    correct_count = 0

    if args.subsample_count:
        q_info_list = q_info_list[:args.subsample_count]  # mostly for testing

    # make requests to paraphrase a bunch of models
    para_requests = []
    for info in q_info_list:
        # Check answer
        answer = info['model_response']
        label = info['answer']
        if re.search(ANSWER_REG, answer) is not None:
            # Does the answer match? Yes or no...
            if ANSWER_PREFIX.format(label) in answer:
                # Correct case; we'll 
                correct_count += 1
                para_req = copy.deepcopy(info)
                para_req['paras_list'] = None
                para_req['num_paras'] = args.para_tries
                para_requests.append(para_req)
        else:
            # print("Answer noncompliant: " + answer)
            noncomply_count += 1

    acc = correct_count / len(q_info_list)
    print("Accuracy: " + str(acc))
    print("Count noncompliant: " + str(noncomply_count))
    print("Total number responses parsed: " + str(len(q_info_list)))

    # split up into the proper number of batches.
    batchcount = args.batches
    print("Building batches...")
    batch_sizes = int(len(para_requests) / batchcount)
    for j in range(batchcount):
        
        if j == batchcount - 1:
            # Grab remainder.
            segment = para_requests[j*batch_sizes:]
        else:
            segment = para_requests[j*batch_sizes:(j+1)*batch_sizes]

        with open(os.path.join(args.batch_create_path, "para_batch_req" + str(j) + ".pickle"), 'wb') as f2:
            pickle.dump(segment, f2)
    
# For managing batched responses.
def strip_eot_start(id_list, eot_token_id):
    start_idx = 0
    for idx in range(len(id_list)):
        if id_list[idx] != eot_token_id:
            start_idx = idx
            break

    return id_list[start_idx:]

def eval_q_batch(pipeline, orig_q, para_list):
    verify_prompt = ("Do sentence 1 and sentence 2 have extremely different meanings? "
        "Remember that some sentences that are phrased differently have the same meaning. "
        "The sentences may appear different, but the wording may actually imply a very similar meaning. "
        "For your answer, please state either \"yes.\" or \"no.\" at the start of your answer."
        "\nSentence 1: '{q1}'"
        "\nSentence 2: '{q2}'\n")
    
    prompt_lengths = []
    promptlist = []
    for test_para in para_list:
        # Get tokenized length.
        verify_msgs = [{"role": "user", "content": verify_prompt.format(q1=orig_q, q2=test_para)}]

        prompt = pipeline.tokenizer.apply_chat_template(verify_msgs, tokenize=False, add_generation_prompt=True)
        encoded_p = pipeline.tokenizer(prompt, padding=False)
        prompt_lengths.append(len(encoded_p['input_ids']))
        promptlist.append(prompt)

    # print("Model Prompt~~~~~~~~~~~~~~")
    # print(prompt)

    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    old_padding = pipeline.tokenizer.padding_side
    pipeline.tokenizer.padding_side = 'left'
    batched_verify_request = pipeline.tokenizer(promptlist, return_tensors="pt",
                                                padding=True)  # Keeping padding off for now.

    # No gradients!
    with torch.no_grad():
        # Put on proper device?
        # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
        batched_verify_request.to(DEVICE_STR)

        generated_output = pipeline.model.generate(**batched_verify_request, do_sample=False, temperature=GEN_TEMP,
                                                max_new_tokens=EVAL_NEW_TOKENS,
                                                return_dict_in_generate=True, output_scores=True, output_logits=True,
                                                pad_token_id=pipeline.tokenizer.eos_token_id)
        
        generated_output = generated_output.sequences

        result_list = []
        for idx in range(generated_output.shape[0]):
            out_ids = generated_output[idx].cpu().detach().numpy().tolist()
            out_ids = strip_eot_start(out_ids, pipeline.tokenizer.eos_token_id)
            out_ids = out_ids[prompt_lengths[idx]:]

            single_response = pipeline.tokenizer.decode(out_ids)
            single_response = single_response.lower()

            # print("idx: " + str(idx) + " " + single_response)

            if re.search("^yes", single_response):  # Begin is "^yes"
                # yes
                # print("Parsed yes")
                # return True
                result_list.append(False)
            elif re.search("^no", single_response):  # Begin is "^no"
                # no
                # print("Parsed no")
                # return False
                result_list.append(True)
            else:
                # print("Could not generate a valid response!")
                result_list.append(None)

    # Restore old padding setting
    pipeline.tokenizer.padding_side = old_padding
    
    return result_list


def strip_eot(id_list, prompt_len, eot_token_id):
    return list(itertools.filterfalse(lambda x: x == eot_token_id, id_list[prompt_len:]))


def paraphrase_question_batch(pipeline, q1, attack_num, temp, max_new_toks=MAX_PARAPHRASE_TOKENS):
    # Paraphrase a question.
    # Multiple tries may be required.
    # Return (newquestion, num_retries). newquestion may be null if we exceeded the maximum number of retries.

    paraphrase_prompt = ("Please paraphrase the following question by substituting words or phrases. "
                     "The paraphrased question should have the same meaning as the original. "
                     "Your response should be surrounded in quotes like the following format: \'<new question>\'.\nOriginal Question: \'{}\'\n")
    
    char_len_proportion = 0.5  # Paraphrase must be within +/- this proportion of the length of the original. Rough metric to remove incorrect responses.

    paraphrase_msgs = [{"role": "user", "content": paraphrase_prompt.format(q1)}]  # question_no_choices

    prompt = pipeline.tokenizer.apply_chat_template(paraphrase_msgs, tokenize=False, add_generation_prompt=True)

    promptlist = [prompt] * attack_num  # 

    # Adjust tokenizer to support padding
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
    batched_paraphrase_request = pipeline.tokenizer(promptlist, return_tensors="pt", padding=True) # Turning padding off for now

    # Same for all of them.
    prompt_length = batched_paraphrase_request['input_ids'].shape[1]

    # Put on proper device?
    # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
    batched_paraphrase_request.to(DEVICE_STR)


    generated_output = pipeline.model.generate(**batched_paraphrase_request, do_sample=True, temperature=temp, max_new_tokens=max_new_toks,
                                        return_dict_in_generate=True, output_scores=True, output_logits=True, pad_token_id=pipeline.tokenizer.eos_token_id)

    generated_output = generated_output.sequences

    para_responses = []
    for idx in range(generated_output.shape[0]):
        # Strip eos tokens.
        out_ids = generated_output[idx].cpu().detach().numpy().tolist()
        out_ids = strip_eot(out_ids, prompt_length, pipeline.tokenizer.eos_token_id)

        single_response = pipeline.tokenizer.decode(out_ids)

        # Parse for response
        all_matches = re.findall("[\"\'].*[\"\']", single_response)  # Switching to single double quotes
        # all_matches = all_matches[1:]  # Exclude first match; that was the original question.

        # The llm tends to put many extra quotes around stuff, chaotically. We look for both quote cases.

        # Just look at the last match.
        if len(all_matches) == 0:
            result = None
        else:
            result = all_matches[-1]

        if result is not None:
            para_q = result
            para_q = para_q.strip("\"\'")
            # Further checks
            # print("Para_q: " + str(para_q))

            # is_paraphrase = eval_q(pipeline, q1, para_q)
            is_identical = (q1 == para_q)
            is_nonempty = len(para_q) > 0
            is_proportion = (len(para_q) > len(q1) - char_len_proportion*len(q1)) and (len(para_q) < len(q1) + char_len_proportion*len(q1))

            # print("Orig len: " + str(len(question_no_choices)))
            # print("Para len: " + str(len(para_q)))

            # print("Is paraphrase?: " + str(is_paraphrase))
            # print("Is identical?: " + str(is_identical))
            # print("Is nonempty?: " + str(is_nonempty))
            # print("Meets proportion requirements?: " + str(is_proportion))

            if (not is_identical) and is_nonempty and is_proportion:
                para_responses.append(para_q)
        # else:
        #     # parse failure in this case, silent for now.

    del generated_output
    del batched_paraphrase_request

    return para_responses

def para_batch(args, api_key):

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

    # Data structure:
        # q_info = {
    #     'uuid': str(uuid.uuid4()),  # Nearly guaranteed unique.
    #     'subject': subject,
    #     'questio_only': data_df.iloc[q_idx, 0],
    #     'question': format_question_str,
    #     'answer': answer_label,
    #     'model_query': format_question_str,
    #     'param_set': param_set,
    #     'model_response': None
    #     'paras_list'
    #     'num_paras'
    # }

    for idx, para_req in enumerate(batch_set):
        if (idx + 1) % PRINT_INTERVAL == 0:
            print("Paraphrase: " + str(idx + 1) + "/" + str(len(batch_set)))
            # Intermediate saving.
            with open(args.batch_result_path, 'wb') as f:
                pickle.dump(batch_set, f)  # This is modified by this process.

        # paraphrase several times.
        paraphrase_candidates = paraphrase_question_batch(pipeline, para_req['questio_only'], para_req['num_paras'], GEN_TEMP, max_new_toks=MAX_PARAPHRASE_TOKENS)
        para_req['paras_list'] = paraphrase_candidates
        para_req['num_paras'] = len(paraphrase_candidates)

    with open(args.batch_result_path, 'wb') as f:
        pickle.dump(batch_set, f)  # This is modified by this process.

# For filtering with LLM evaluator.
def eval_set(args, api_key):

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

    starttime = datetime.datetime.now()
    print("LLM paraphrase eval: " + str(starttime))

    loadpath = args.eval_batch_file
    savepath = args.eval_batch_result_path

    # We'll add a 'evaled_paras_list' attribute to each question
    with open(loadpath, 'rb') as f:
        q_infos = pickle.load(f)

    for idx, q_info in enumerate(q_infos):
        if (idx + 1) % PRINT_INTERVAL == 0:
            print("Evaluating: " + str(idx + 1) + "/" + str(len(q_infos)))
            elapsed = datetime.datetime.now() - starttime
            print("Elapsed seconds: " + str(elapsed.total_seconds()))
            # Intermediate saving.
            with open(savepath, 'wb') as f2:
                pickle.dump(q_infos, f2)

        if(len(q_info['paras_list']) != 0):

            evaluations = eval_q_batch(pipeline, q_info['questio_only'], q_info['paras_list'])
            q_info['paras_list_evals'] = evaluations

            true_paras = []
            for idx2 in range(len(q_info['paras_list'])):
                if evaluations[idx2] is not None:
                    if evaluations[idx2]:
                        true_paras.append(q_info['paras_list'][idx2])

            q_info['evaled_paras_list'] = true_paras
        else:
            q_info['paras_list_evals'] = None
            q_info['evaled_paras_list'] = []
        
    with open(savepath, 'wb') as f2:
        pickle.dump(q_infos, f2)

    endtime = datetime.datetime.now()
    print("LLM paraphrase Finished: " + str(endtime))
    print("Elapsed seconds: " + str((endtime - starttime).total_seconds()))

def result_stats(args):

    results_dir = args.results_dir
    prefix = args.results_prefix

    file_candidates = os.listdir(results_dir)

    q_list = []
    for file in file_candidates:
        if re.match(prefix + "[0-9]+", file):
            with open(os.path.join(results_dir, file), 'rb') as f:
                q_list = q_list + pickle.load(f)

    # parse questions.

    # Total number of paraphrases
    num_paras_list = []
    num_failures = 0
    for q_info in q_list:
        num_paras_list.append(q_info['num_paras'])
        if(q_info['num_paras'] == 0):
            num_failures += 1


    print("Random example: ")
    rand_entry = random.choice(q_list)
    print("Original question: " + rand_entry['questio_only'])
    print("Paraphrases: ")
    for p in rand_entry['paras_list']:
        print(p)
    
    print("Total number of questions: " + str(len(q_list)))
    print("Total number of generated paraphrases: " + str(sum(num_paras_list)))
    print("For 20 paraphrase attempts: ")
    print(" Min: " + str(np.min(num_paras_list)))
    print(" Mean: " + str(np.mean(num_paras_list)))
    print(" Max: " + str(np.max(num_paras_list)))
    # min/mean/max number of successful paraphrases
    print("~~~~~~~~~~~~~~~")
    print("Number of questions with no paraphrases (noncompliance): " + str(num_failures))
    print("Proportion of questions with no paraphrases: " + str(num_failures / len(q_list)))


def main():
    print("Start")
    parser = argparse.ArgumentParser()
    # Build batches
    parser.add_argument("--batches", "-b", type=int)
    parser.add_argument("--subsample_count", "-n", type=int, default=None)
    parser.add_argument("--batch_create_path", "-o", type=str)
    parser.add_argument("--response_file", "-i", type=str)

    # More arguments
    # How many paraphrases to generate
    parser.add_argument("--para_tries", "-t", type=int, default=20)

    # Process batch
    parser.add_argument("--batch_file", "-f", type=str)
    parser.add_argument("--batch_result_path", "-r", type=str)

    # TODO: Add option for evaluator.
    parser.add_argument("--eval_batch_file", type=str)
    parser.add_argument("--eval_batch_result_path", type=str)

    # See results
    parser.add_argument("--results_dir", "-p", type=str)
    parser.add_argument("--results_prefix", "-e", type=str)


    print("Parsing args...")
    args = parser.parse_args()

    if args.results_dir and args.results_prefix:
        result_stats(args)
    elif (args.batch_file and args.batch_result_path) or (args.eval_batch_file and args.eval_batch_result_path):  # Process single batch file.
        

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

        if args.eval_batch_file and args.eval_batch_result_path:
            print("Running evaluation batch")
            eval_set(args, api_key=API_KEY)
        else:
            print("Running paraphrasing batch")
            para_batch(args, api_key=API_KEY)

    else:  # assume we're making a batch.
        if args.batches and args.batch_create_path and args.response_file:
            print("Building batches")
            build_batches(args)
        else:
            print("Missing arguments!")

if __name__ == "__main__":
    main()
