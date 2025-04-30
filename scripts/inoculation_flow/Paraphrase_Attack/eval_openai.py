
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
from matplotlib import pyplot as plt
import io
import json

import transformers
import torch
import openai

from transformers import LlamaForCausalLM, LlamaTokenizerFast

# Evaluate openai with the current paraphrasing strategy.
# Batch up 20 paras per Q, see ASR.

KEYPATH = ""
with open(KEYPATH, 'r') as f:
    API_KEY = f.readline()
    API_KEY = API_KEY.rstrip('\n')

OPENAI_KEYPATH = ""
with open(OPENAI_KEYPATH, 'r') as f:
    OPENAI_API_KEY = f.readline()
    OPENAI_API_KEY = OPENAI_API_KEY.rstrip('\n')

# LLaMA model links:
# https://huggingface.co/meta-llama/Llama-2-7b-chat/tree/main
# Nevermind, use this similar model instead:
# https://huggingface.co/meta-llama/Llama-2-7b-hf
# Example code for using transformers with the model. I used this as a starting point:
# https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed

# Try a bigger model...
# TEST_MODEL = "Llama-3.1-8B-Instruct"
# MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"  # Other model does not have a chat template; no chat support?

LOCAL_MODEL = "Llama-3.2-3B-Instruct" # This model does the paraphrasing.
MODEL_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # Other model does not have a chat template; no chat support?

GEN_TEMP = 1.0

LOCAL_EOT_STR = "<|eot_id|>"

LOCAL_DEVICE_STR = 'cuda'


# Training sets
PAWS_WIKI_PATH = '/home/jlim/MS_Project/validator_validation/paws_wiki_labeled_final/final/train.tsv'
PAWS_QQP_PATH = '/home/jlim/MS_Project/validator_validation/paws_qqp/output/train.tsv'

# PAWS_WIKI_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_wiki_labeled_final/final/train.tsv'
# PAWS_QQP_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_qqp/output/train.tsv'


# For short answers
MAX_NEW_TOKENS = 10

# For paraphrasing.
MAX_PARAPHRASE_TOKENS = 1000  # Roughly 2x the number of tokens for the longest question.

CHOICES = ["A", "B", "C", "D"]

BATCH_SIZE = 1  # one at a time for now. Avoids padding issues.


DATA_DIR = "/home/jlim/MS_Project/MMLU_Baseline/data"

ANSWER_STRING_MAP = {'0': "A", '1': "B", '2': "C", '3': "D"}

# print("Data directory: " + str(DATA_DIR))

PARSE_RETRIES = 1 # 5 usually. 1 for no sampling.

# Function Defs
def eval_q(pipeline, q1, q2):
    # return the label that the model answers for a given prompt.
    # True means paraphrase, False means no paraphrase.

    # More testing - best so far
    verify_prompt = ("Do sentence 1 and sentence 2 have extremely different meanings? "
        "Remember that some sentences that are phrased differently have the same meaning. "
        "The sentences may appear different, but the wording may actually imply a very similar meaning. "
        "For your answer, please state either \"yes.\" or \"no.\" at the start of your answer."
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

        # No gradients!
        with torch.no_grad():
            # Put on proper device?
            # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
            encoded_verify_request.to(LOCAL_DEVICE_STR)

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

        # if re.search(re.escape("yes.<|eot_id|>") + "$", generated_output):  # Begin is "^yes"
        #     # yes
        #     # print("Parsed yes")
        #     return True
        # elif re.search(re.escape("no.<|eot_id|>") + "$", generated_output):  # Begin is "^no"
        #     # no
        #     # print("Parsed no")
        #     return False

        else:
            fail_count += 1
            print("Response validation failure. Fail count: " + str(fail_count))
            # print("Failing Validation response:")
            # print(generated_output)
            if fail_count > PARSE_RETRIES:
                print("Could not generate a valid response!")
                return None
            
MAX_PARA_RETRIES = 20

PARA_TEMP = 1.2  # default: 1.0

# For paraphrasing.
MAX_PARAPHRASE_TOKENS = 1000  # Roughly 2x the number of tokens for the longest question.
            
def paraphrase_question(pipeline, q1, max_retries=MAX_PARA_RETRIES, temp=PARA_TEMP, max_new_toks=MAX_PARAPHRASE_TOKENS):
    # Paraphrase a question.
    # Multiple tries may be required.
    # Return (newquestion, num_retries). newquestion may be null if we exceeded the maximum number of retries.

    paraphrase_prompt = ("Please paraphrase the following question by substituting words or phrases. "
                     "The paraphrased question should have the same meaning as the original. "
                     "Your response should be surrounded in quotes like the following format: \'<new question>\'.\nOriginal Question: \'{}\'\n")
    
    char_len_proportion = 0.5  # Paraphrase must be within +/- this proportion of the length of the original. Rough metric to remove incorrect responses.

    paraphrase_msgs = [{"role": "user", "content": paraphrase_prompt.format(q1)}]  # question_no_choices

    successful = False
    parse_fail_count = 0
    para_q = None

    para_count = 0

    prompt = pipeline.tokenizer.apply_chat_template(paraphrase_msgs, tokenize=False, add_generation_prompt=True)
    encoded_paraphrase_request = pipeline.tokenizer(prompt, return_tensors="pt", padding=False) # Turning padding off for now

    # Put on proper device?
    # Inspiration: https://discuss.huggingface.co/t/device-map-auto-with-error-expected-all-tensors-to-be-on-the-same-device/31938/6
    encoded_paraphrase_request.to(LOCAL_DEVICE_STR)

    while not successful:
        para_count += 1
        # print("Paraphrase attempt: " + str(para_count))
        if para_count > max_retries:
            return (None, para_count)

        generated_output = pipeline.model.generate(**encoded_paraphrase_request, do_sample=True, temperature=temp, max_new_tokens=max_new_toks,
                                            return_dict_in_generate=True, output_scores=True, output_logits=True, pad_token_id=pipeline.tokenizer.eos_token_id)

        # generated_output = generated_output.sequences[0][-MAX_PARAPHRASE_TOKENS:]
        generated_output = generated_output.sequences[0][encoded_paraphrase_request['input_ids'].shape[1]:]

        # Detokenize
        generated_output = pipeline.tokenizer.decode(generated_output)

        # print("Model output ~~~~~~~~~~~~~~~~~~~")
        # print(generated_output)
        # continue

        # result = re.search("\"\".*\"\"", generated_output)

        # The prompt has a string that matches this. Ignore this by getting only the last match!
        # all_matches = re.findall("\"\".*\"\"", generated_output)
        all_matches = re.findall("[\"\'].*[\"\']", generated_output)  # Switching to single double quotes
        # all_matches = all_matches[1:]  # Exclude first match; that was the original question.

        # The llm tends to put many extra quotes around stuff, chaotically. We look for both quote cases.

        # Just look at the last match.
        if len(all_matches) == 0:
            result = None
        else:
            result = all_matches[-1]

        if result is not None:
            # para_q = generated_output[result.start():result.end()]
            para_q = result
            para_q = para_q.strip("\"\'")

            # print("Paraphrase response:")
            # print(generated_output)
            # print("~~~~~~~~~~~~~~~~~~~~~")

            # print("Number of parse retries: " + str(fail_count))
            # print("Paraphrased question; original:")
            # print(para_q)
            # print(question_no_choices)

            # print("Comparing to: " + str(question_no_choices))
            is_paraphrase = eval_q(pipeline, q1, para_q)
            is_identical = (q1 == para_q)
            is_nonempty = len(para_q) > 0
            is_proportion = (len(para_q) > len(q1) - char_len_proportion*len(q1)) and (len(para_q) < len(q1) + char_len_proportion*len(q1))

            # print("Orig len: " + str(len(question_no_choices)))
            # print("Para len: " + str(len(para_q)))

            # print("Is paraphrase?: " + str(is_paraphrase))
            # print("Is identical?: " + str(is_identical))
            # print("Is nonempty?: " + str(is_nonempty))
            # print("Meets proportion requirements?: " + str(is_proportion))

            if is_paraphrase and (not is_identical) and is_nonempty and is_proportion:
                # Keep tensor memory clear!
                del encoded_paraphrase_request
                successful = True

                return (para_q, para_count)
                

        else:
            parse_fail_count += 1
            # print("Failing Paraphrase response:")
            # print(generated_output)
            if parse_fail_count > PARSE_RETRIES:
                # Fallback to not paraphrasing?
                print("Could not parse a valid paraphrase response!")
                # num_rephrases[-1] = -1
                
                return (None, para_count)

    return (None, para_count)

POLL_PERIOD = 3

def batch_eval(chat_completion_param_set: list, openai_client):
    # Use openai's batch api; send a large number of chat completion requests, then block for a response.

    batch_file = io.BytesIO()

    # num_completions = len(chat_completion_param_set)

    # make an in-memory json file
    for idx, params in enumerate(chat_completion_param_set):
        # Add custom id, because return order is not guaranteed
        request_obj = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": params,
        }

        # Assuming utf-8????
        dumpstr = json.dumps(request_obj).encode()
        # json.dump(params, batch_file)
        batch_file.write(dumpstr)

        batch_file.write("\n".encode())  # jsonl format

    batch_file.seek(0)
    # print("File contents: ")
    # print(batch_file.read())

    # Reference: https://platform.openai.com/docs/api-reference/files?lang=python
    # https://platform.openai.com/docs/guides/batch

    # 0: List current files.
    # files_list = openai_client.files.list()
    #
    # # When we need to clear files
    # for f in files_list:
    #     delete_stat = openai_client.files.delete(f.id)
    # files_list = openai_client.files.list()

    # 1: Upload batch file
    uploaded_file = openai_client.files.create(file=batch_file, purpose="batch")
    # Debugging
    # uploaded_file = openai_client.files.create(None)

    fid = uploaded_file.id

    # 2: Tell OpenAI to run the particular batch file
    # NOTE: Correct endpoint?
    batch = openai_client.batches.create(input_file_id=fid,
                                         endpoint="/v1/chat/completions", # /v1/completions
                                         completion_window='24h')
    batch_id = batch.id

    # JL - having weird issues with batches not starting. So adding a simple retry loop for stability?
    # For whatever reason
    started = False
    retry_count = 0
    while not started:
        time.sleep(POLL_PERIOD)
        batch_update = openai_client.batches.retrieve(batch_id)

        if batch_update.status != 'in_progress':
            if batch_update.status == 'failed':
                if type(batch_update.errors.data[0]) == openai.types.batch_error.BatchError:
                    retry_count += 1
                    print("Retrying, count: " + str(retry_count))

                    batch = openai_client.batches.create(input_file_id=fid,
                                                         endpoint="/v1/chat/completions",  # /v1/completions
                                                         completion_window='24h')
                    batch_id = batch.id
                else:
                    raise Exception("Batch ended with status: " + batch_update.status)
        else:
            started = True

    print("Batch started successfully")
    # 3: Wait for results (can poll periodically I guess)
    output_file_id = None
    finished = False
    while not finished:
        time.sleep(POLL_PERIOD)  # Wait period, so as to not spam polling...
        batch_update = openai_client.batches.retrieve(batch_id)

        # check for all terminal states
        status = batch_update.status  # error, 'method' parameter...
        print("Batch status: " + str(status))
        if status == 'failed' or status == 'expired' or status == 'cancelling' or status == 'cancelled':
            # delete_status = openai_client.files.delete(fid)  # Clean up failed batch file.
            raise Exception("Batch ended with status: " + status)
        elif status == 'completed':
            finished = True
            output_file_id = batch_update.output_file_id


    # 4: When results are ready, pull them down.
    response_file = openai_client.files.content(output_file_id)

    # file format notes:
    # look at response_file.text in example.jsonl, to format parsing the response
    # Parse response
    response_lines = response_file.text.split("\n")
    responses = [json.loads(x) for x in response_lines if len(x) > 0]

    # 5: Clean up files created on openai's api.
    delete_status = openai_client.files.delete(fid)
    # 5.5: Delete output file too.
    delete_status2 = openai_client.files.delete(output_file_id)

    # 6: Use custom ids to re-order and return responses
    responses = sorted(responses, key=lambda x: x["custom_id"])

    response_strs = [x["response"]["body"]["choices"][0]["message"]["content"] for x in responses]

    # list of top logprobs: x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    # top_tokens_lists = [[y["token"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]
    # top_logprobs_lists = [[y["logprob"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]

    # print("done")
    # files_list = openai_client.files.list()
    # print("Files remaining on openAI api: " + str(len(files_list.data)))

    return response_strs

RESPONSE_MAX_NEW_TOKENS = 240  # Not sure how long this should be?

ANSWER_TEMP = 1

def build_question_batch(model_identifier, question_list, max_new_tokens=RESPONSE_MAX_NEW_TOKENS):

    prompt_string = ("Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\n"
                        "Your response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.")

    batch_param_sets = []

    for q in question_list:

        # N
        # msgs = [{"role": "system", "content": train_prompt},
        #                 {"role": "user", "content": prompt_end}]
        # No system role; we don't have few-shot examples here.
        msgs = [{"role": "user", "content": prompt_string.format(q)}]

        param_set = {
            'model': model_identifier,
            'messages': msgs,
            'logprobs': False,  # False for now.
            'max_tokens': RESPONSE_MAX_NEW_TOKENS,
            'temperature': ANSWER_TEMP,
            # 'top_logprobs': 20
        }

        batch_param_sets.append(param_set)

    return batch_param_sets


def format_para_question_single(base_q, df, idx, include_answer=True):
    prompt = base_q
    k = df.shape[1] - 2  # answer index.
    for j in range(k):
        prompt += "\n{}. {}".format(CHOICES[j], df.iloc[idx, j+1])

    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

ATTACK_TRIES = 10  # Try this many attacks on a question at once.

UNIQUENESS_RETRIES = 10  # Try up to additional this many times to get the desired number of unique paraphrases.

def main():

    # Old model first
    OPENAI_TEST_MODEL = "gpt-3.5-turbo-0125"

    # Probably the best comparison
    # OPENAI_TEST_MODEL = "gpt-4o-mini-2024-07-18"

    ANSWER_PREFIX = "The best answer is {}"
    
    # Load MMLU dev set
    # read all questions
    mmlu_subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(os.path.join(DATA_DIR, "test")) if "_test.csv" in f])

    dev_df = None
    for subject in mmlu_subjects:
        # TODO: Separate by subject?
        if dev_df is None:
            dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev", subject + "_dev.csv"), header=None)
        else:
            dev_df = pd.concat([dev_df, pd.read_csv(os.path.join(DATA_DIR, "dev", subject + "_dev.csv"), header=None)])
            
    num_qs = dev_df.shape[0]
    print("Dev dataset size: " + str(num_qs))

    pipeline = None
    print("Initializing model & pipeline...")
    # Model, tokenizer, pipeline init
    device_map_setting = LOCAL_DEVICE_STR  # 'auto'  # 'cuda'

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

    # Set up openai api stuff.
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    print("Building unmodified question batch...")

    q_orig_set = []
    q_only_set = []

    # Build batch
    for q_idx in range(dev_df.shape[0]):
        orig_question_str = format_para_question_single(dev_df.iloc[q_idx, 0], dev_df, q_idx, include_answer=False)

        q_orig_set.append(orig_question_str)
        q_only_set.append(dev_df.iloc[q_idx, 0])
        
        # print("Batching Question: " + str(q_idx+1) + "/" + str(dev_df.shape[0]))
    
    mdl_requests = build_question_batch(OPENAI_TEST_MODEL, q_orig_set, max_new_tokens=RESPONSE_MAX_NEW_TOKENS)

    print("Submitting question batch...")
    responses = batch_eval(mdl_requests, openai_client)
    print("Batch responses retrieved.")

    q_incorrect_idxs = []

    # Get 
    for r_idx, answer in enumerate(responses):
        label = dev_df.iloc[r_idx, dev_df.shape[1]-1]  # Last column.
        
        # TODO: Stats on noncompliant responses?

        # Does the answer match? Yes or no...
        if ANSWER_PREFIX.format(label) in answer:
            q_incorrect_idxs.append(r_idx)

    num_initial_correct = len(q_orig_set) - len(q_incorrect_idxs)
    initial_accuracy = num_initial_correct / len(q_orig_set)
    print("Accuracy: " + str(initial_accuracy))

    # Find incorrectly answered questions. Paraphrase these.
    print("Paraphrasing correct questions; Question count: " + str(len(q_incorrect_idxs)))

    print("Paraphrases per question: " + str(ATTACK_TRIES))

    # So batch request length is ATTACK_TRIES * len(q_incorrect_idxs)

    num_paras_mapping = []  # remember how many paraphrases were creatd.
    paras_batch_set = []
    for r_index in q_incorrect_idxs:
        question = dev_df.iloc[r_index, 0]

        para_set = []
        for a in range(ATTACK_TRIES + UNIQUENESS_RETRIES):
            # Use llama to paraphrase

            # NOTE: Exclude duplicate paraphrases...
            # JL NOTE: Using simple reject/retry technique.

            para_q, tries = paraphrase_question(pipeline, question, max_retries=MAX_PARA_RETRIES, temp=PARA_TEMP, max_new_toks=MAX_PARAPHRASE_TOKENS)

            if para_q is not None:
                if para_q not in para_set:
                    para_set.append(para_q)
                    if len(para_set) == ATTACK_TRIES:
                        num_paras_mapping == ATTACK_TRIES
                        break

        num_paras_mapping.append(len(para_set))

        # Format questions.
        paras_batch_set + [format_para_question_single(x, dev_df, r_index, include_answer=False) for x in para_set]


    para_mdl_requests = build_question_batch(OPENAI_TEST_MODEL, paras_batch_set, max_new_tokens=RESPONSE_MAX_NEW_TOKENS)

    print("Submitting question batch...")
    responses = batch_eval(para_mdl_requests, openai_client)
    print("Batch responses retrieved.")

    # Parse using num_paras_mapping as well

    # Calculate ASR.
    # Save responses, paraphrased Qs?

    # Just try ASR first...

    count_incorrect = 0

    for idx, mapping in enumerate(num_paras_mapping):
        base_idx = 0
        for j in range(0, idx):
            base_idx += sum(num_paras_mapping[j])

        for subidx in range(mapping):
            # Actually need to do cumulative sum
            full_idx = base_idx + subidx
            # Answer correct or not?

            label = dev_df.iloc[full_idx, dev_df.shape[1]-1]  # Last column.
        
            # TODO: Stats on noncompliant responses?
            answer = responses[full_idx]

            # Does the answer match? Yes or no...
            if not (ANSWER_PREFIX.format(label) in answer):
                count_incorrect += 1
                break  # Attack successful, skip to next paraphrase set.

    asr = count_incorrect / num_initial_correct
    print("Attack Success Proportion: " + str(asr))

if __name__ == "__main__":
    main()