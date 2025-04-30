# Submit & manage batch of MMLU data from openai's gpt4o model...

# Setup
import argparse
import os, sys
import uuid

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
import pickle

import transformers
import torch
import openai

OPENAI_KEYPATH = "/home/jeremy/Documents/WPI_MS/Q_Inoculate/OpenAI_Api_free"
with open(OPENAI_KEYPATH, 'r') as f:
    OPENAI_API_KEY = f.readline()
    OPENAI_API_KEY = OPENAI_API_KEY.rstrip('\n')

GEN_TEMP = 1.0


# PAWS_WIKI_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_wiki_labeled_final/final/train.tsv'
# PAWS_QQP_PATH = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/validator_validation/paws_qqp/output/train.tsv'


# For short answers
MAX_NEW_TOKENS = 20

GEN_TEMP = 1.0

# For paraphrasing.
MAX_PARAPHRASE_TOKENS = 1000  # Roughly 2x the number of tokens for the longest question.

CHOICES = ["A", "B", "C", "D"]


DATA_DIR = "/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/MMLU_Data/data"

ANSWER_STRING_MAP = {'0': "A", '1': "B", '2': "C", '3': "D"}

OPENAI_TEST_MODEL = "gpt-4o-mini-2024-07-18"

ANSWER_PREFIX = "The best answer is {}"

ANSWER_REG = "The best answer is [ABCD]"

MAX_REQUEST_SIZE = 50000

def send_batch(chat_completion_param_set: list, openai_client):
    # Use openai's batch api; send a large number of chat completion requests, then block for a response.

    batch_file = io.BytesIO()

    # num_completions = len(chat_completion_param_set)

    # make an in-memory json file
    for idx, question in enumerate(chat_completion_param_set):
        # Add custom id, because return order is not guaranteed
        request_obj = {
            "custom_id": question['uuid'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": question['param_set'],
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

    # return batch id, file id.
    return batch_id, fid

    # 4: When results are ready, pull them down.
    # response_file = openai_client.files.content(output_file_id)
    #
    # # file format notes:
    # # look at response_file.text in example.jsonl, to format parsing the response
    # # Parse response
    # response_lines = response_file.text.split("\n")
    # responses = [json.loads(x) for x in response_lines if len(x) > 0]
    #
    # # 5: Clean up files created on openai's api.
    # delete_status = openai_client.files.delete(fid)
    # # 5.5: Delete output file too.
    # delete_status2 = openai_client.files.delete(output_file_id)
    #
    # # 6: Use custom ids to re-order and return responses
    # responses = sorted(responses, key=lambda x: x["custom_id"])
    #
    # response_strs = [x["response"]["body"]["choices"][0]["message"]["content"] for x in responses]

    # list of top logprobs: x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    # top_tokens_lists = [[y["token"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]
    # top_logprobs_lists = [[y["logprob"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]

    # print("done")
    # files_list = openai_client.files.list()
    # print("Files remaining on openAI api: " + str(len(files_list.data)))

    # return response_strs

def process_batch(questions_pickle_path, api_response_file, output_file):
    # Parse a batch we got back from openai

    # 4: When results are ready, pull them down.
    # response_file = openai_client.files.content(output_file_id)

    with open(api_response_file, 'r') as f:
        response_lines = f.readlines()

    # response_lines = response_file.text.split("\n")

    responses = [json.loads(x) for x in response_lines if len(x) > 0]

    # # 6: Use custom ids to re-order and return responses
    # responses = sorted(responses, key=lambda x: x["custom_id"])
    #

    # essentially, join responses with existing questions record.

    # response_strs = [x["response"]["body"]["choices"][0]["message"]["content"] for x in responses]

    with open(questions_pickle_path, 'rb') as f2:
        batch_set = pickle.load(f2)

    # Remap data structures by uuid.
    # Should be a bit faster.
    uuid_to_q = {}
    for q in batch_set:
        uuid_to_q[q['uuid']] = q
        if 'evaled_qinfos' in q:
            for p_q in q['evaled_qinfos']:
                uuid_to_q[p_q['uuid']] = p_q

    count_unlinked = 0
    for idx, resp in enumerate(responses):
        if idx % 100 == 0:
            print(str(idx) + "/" + str(len(responses)))
        resp_id = resp["custom_id"]
        response_str = resp["response"]["body"]["choices"][0]["message"]["content"]
        # Not efficient, but whatever.
        # found = False
        # for q in batch_set:
        #     if q['uuid'] == resp_id:
        #         found = True
        #         q['model_response'] = response_str
        #         break
        #     elif 'evaled_qinfos' in q:
        #         for p_q in q['evaled_qinfos']:
        #             if p_q['uuid'] == resp_id:
        #                 found = True
        #                 p_q['model_response'] = response_str
        #                 break
        #
        #         if found:
        #             break
        #
        # if not found:
        #     print("Could not link response; UUID: " + str(resp_id))
        if resp_id in uuid_to_q:
            q_record = uuid_to_q[resp_id]
            q_record['model_response'] = response_str
        else:
            print("Could not link response; UUID: " + str(resp_id))
            count_unlinked += 1

    print("Count unlinked: " + str(count_unlinked))

    with open(output_file, 'wb') as f3:
        pickle.dump(batch_set, f3)


def format_para_question_single(base_q, df, idx, include_answer=True):
    prompt = base_q
    k = df.shape[1] - 2  # answer index.
    for j in range(k):
        prompt += "\n{}. {}".format(CHOICES[j], df.iloc[idx, j+1])

    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def make_request(question_path, prefix):
    # Do dev set
    set_prefix = prefix

    mmlu_subjects = sorted(
        [f.split("_test.csv")[0] for f in os.listdir(os.path.join(DATA_DIR, "test")) if "_test.csv" in f])

    prompt_string = ("Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\n"
                "Your response should begin with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.\n"
                "If your response does not follow the described format, it will be incorrect.")

    data_df = None

    num_qs = 0
    subject_sets = []
    for subject in mmlu_subjects:
        # Separating by subject.
        # if data_df is None:
        #     data_df = pd.read_csv(os.path.join(DATA_DIR, set_prefix, subject + "_" + set_prefix + ".csv"), header=None)
        # else:
        #     data_df = pd.concat([data_df, pd.read_csv(os.path.join(DATA_DIR, set_prefix, subject + "_" + set_prefix + ".csv"), header=None)])

        data_df = pd.read_csv(os.path.join(DATA_DIR, set_prefix, subject + "_" + set_prefix + ".csv"), header=None)

        subject_question_set = []

        num_qs += data_df.shape[0]

        for q_idx in range(data_df.shape[0]):
            format_question_str = format_para_question_single(data_df.iloc[q_idx, 0], data_df, q_idx,
                                                            include_answer=False)
            answer_label = data_df.iloc[q_idx, data_df.shape[1]-1]

            msgs = [{"role": "user", "content": prompt_string.format(format_question_str)}]

            param_set = {
                'model': OPENAI_TEST_MODEL,
                'messages': msgs,
                'logprobs': False,  # False for now.
                # 'max_tokens': RESPONSE_MAX_NEW_TOKENS, # GPT turbo
                'temperature': GEN_TEMP,
                'max_completion_tokens': MAX_NEW_TOKENS,  # o3 mini
                # 'top_logprobs': 20
            }

            q_info = {
                'uuid': str(uuid.uuid4()),  # Nearly guaranteed unique.
                'subject': subject,
                'questio_only': data_df.iloc[q_idx, 0],
                'question': format_question_str,
                'answer': answer_label,
                'model_query': format_question_str,
                'param_set': param_set,
                'model_response': None
            }

            # Sent prompt
            # param_dictionary
            # Response

            subject_question_set.append(q_info)

        subject_sets.append(subject_question_set)

    print(set_prefix + " total dataset size: " + str(num_qs))

    # for now, doing all subjects
    batch_set = []
    for subj in subject_sets:
        batch_set = batch_set + subj

    # Save
    #  "questions_" + set_prefix + ".pickle"
    with open(question_path, 'wb') as f_obj:
        pickle.dump(batch_set, f_obj)

    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Make request.
    batch_id, fid = send_batch(batch_set, openai_client)

    print("Sent batch...")
    print("Batch id: " + str(batch_id))
    print("Fid: " + str(fid))

# Making variant that does not use the batch api. It behaves weirdly slow, inconsistently.
def do_request():
    pass


def request_para_answers(para_input_file, para_output_file):
    with open(para_input_file, 'rb') as f:
        q_infos = pickle.load(f)

    print("Building request to evaluate paraphrases...")

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
    #     'paras_list_evals'
    #     'evaled_paras_list'
    #     'evaled_qinfos': list({
    #         'para_q': str
    #         'param_set' {}
    #         'uuid': str
    #         'model_response': str

    #      })
    # }

    prompt_string = ("Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\n"
                "Your response should begin with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.\n"
                "If your response does not follow the described format, it will be incorrect.")

    # Assign unique UUID to each paraphrase question.
    answer_requests = []


    for q_info in q_infos:

        # Adding a new data structure
        eval_infos = []

        orig_q_len = len(q_info['questio_only'])
        q_body = q_info['question'][orig_q_len:]

        if 'evaled_qinfos' in q_info:
            # Already have a uuid...
            for para_info in q_info['evaled_qinfos']:
                answer_requests.append(para_info)
        else:
            for para_q in q_info['evaled_paras_list']:
                format_question_str = para_q + q_body

                msgs = [{"role": "user", "content": prompt_string.format(format_question_str)}]

                param_set = {
                    'model': OPENAI_TEST_MODEL,
                    'messages': msgs,
                    'logprobs': False,  # False for now.
                    # 'max_tokens': RESPONSE_MAX_NEW_TOKENS, # GPT turbo
                    'temperature': GEN_TEMP,
                    'max_completion_tokens': MAX_NEW_TOKENS,  # o3 mini
                    # 'top_logprobs': 20
                }

                info_dict = {
                    'para_q': para_q,
                    'param_set': param_set,
                    'uuid': str(uuid.uuid4()),
                    'model_response': None,
                }

                eval_infos.append(info_dict)
                answer_requests.append(info_dict)

            q_info['evaled_qinfos'] = eval_infos

    # build a batch to get paraphrased answers for every paraphrase...
    with open(para_output_file, 'wb') as f2:
        pickle.dump(q_infos, f2)

    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    if len(answer_requests) > MAX_REQUEST_SIZE:
        print("Making sub requests...")
        num_requests = int(len(answer_requests) / MAX_REQUEST_SIZE)
        if len(answer_requests) % MAX_REQUEST_SIZE != 0:
            num_requests += 1

        for b in range(num_requests):
            sub_requests = answer_requests[b*MAX_REQUEST_SIZE:(b+1)*MAX_REQUEST_SIZE]

            # Need to redo the first two requests...
            if b != 2:
                batch_id, fid = send_batch(sub_requests, openai_client)
                print("Sub batch: " + str(b))
                print("Sub batch: Number of requests: " + str(len(sub_requests)))
                print("Batch id: " + str(batch_id))
                print("Fid: " + str(fid))

    else:

        # Make request.
        batch_id, fid = send_batch(answer_requests, openai_client)

        print("Number of requests: " + str(len(answer_requests)))
        print("Batch id: " + str(batch_id))
        print("Fid: " + str(fid))

def deduplicate_para_qs(paras_infos):
    # map string to records
    conflict_exists = False
    paras_map = {}
    for p_info in paras_infos:
        if not (p_info['para_q'] in paras_map):
            paras_map[p_info['para_q']] = p_info
        else:
            # check existing record; see if there's an answer conflict
            exist_record = paras_map[p_info['para_q']]

            exist_search = re.search(ANSWER_REG, exist_record['model_response'])
            new_search = re.search(ANSWER_REG, p_info['model_response'])

            # Do they parse to the same answer?
            if exist_search is not None and new_search is not None:
                # Does the answer match? Yes or no...
                if exist_record['model_response'][exist_search.span()[0]:exist_search.span()[1]][-1] != p_info['model_response'][new_search.span()[0]:new_search.span()[1]][-1]:
                    conflict_exists = True
                    # print("Conflict: Model gave different response for exact same question!!!")
                    # print(exist_search.string)
                    # print(new_search.string)
            else:
                conflict_exists = True

            # if exist_record['model_response'] != p_info['model_response']:
            #     print("Conflict: Model gave different response for exact same question!!!")
            #     print(exist_record['model_response'])
            #     print(p_info['model_response'])

    retlist = []
    for key in paras_map.keys():
        retlist.append(paras_map[key])

    return (retlist, conflict_exists)

def check_stats(output_file, question_condition=lambda x: True):
    with open(output_file, 'rb') as f2:
        q_info_list = pickle.load(f2)

    correct_count = 0
    noncomply_count = 0

    one_success_count = 0

    attack_success_counts = []

    count_actually_para = 0

    count_paras_dist = []

    map_paras_freq = {}
    map_paras_success = {}  # Help compute asr vs # of paraphrases.
    map_paras_total_success = {}
    map_paras_total_count = {}
    for a in range(0, 21):
        map_paras_freq[a] = 0
        map_paras_success[a] = 0
        map_paras_total_success[a] = 0
        map_paras_total_count[a] = 0


    # By subject?
    map_qs_by_subject = {}
    map_1success_by_subject = {}
    map_paras_by_subject = {}
    map_total_success_by_subject = {}

    count_paras = 0
    count_paras_success = 0

    # Add a condition; for the question/answers.
    q_info_list = list(filter(question_condition, q_info_list))

    conflicting_output_count = 0

    for info in q_info_list:

        # Compute subject stats.
        if info['subject'] not in map_qs_by_subject:
            map_qs_by_subject[info['subject']] = 0
            map_1success_by_subject[info['subject']] = 0
            map_paras_by_subject[info['subject']] = 0
            map_total_success_by_subject[info['subject']] = 0

        map_qs_by_subject[info['subject']] += 1

        # Check answer
        answer = info['model_response']
        label = info['answer']
        if re.search(ANSWER_REG, answer) is not None:
            # Does the answer match? Yes or no...
            if ANSWER_PREFIX.format(label) in answer:
                # Correct case; we'll
                correct_count += 1

        else:
            # print("Answer noncompliant: " + answer)
            noncomply_count += 1

        # Second level: Analyze paraphrase performance
        if 'evaled_qinfos' in info:
            p_info_list, conflict = deduplicate_para_qs(info['evaled_qinfos'])

            if conflict:
                conflicting_output_count += 1
            success_attacks = 0
            if len(p_info_list) > 0 or not conflict: # We skip questions with duplicate paraphrases and conflicting answers.
                count_actually_para += 1
            count_paras += len(p_info_list)
            count_paras_dist.append(len(p_info_list))

            map_paras_freq[len(p_info_list)] += 1
            map_paras_total_count[len(p_info_list)] += len(p_info_list)
            map_paras_by_subject[info['subject']] += len(p_info_list)

            for p_info in p_info_list:
                # print("Original: " + info['questio_only'])
                # print("Paraphrase: " + p_info['para_q'])
                answer = p_info['model_response']
                # We ignore noncompliant responses.
                if re.search(ANSWER_REG, answer) is not None:
                    # Does the answer match? Yes or no...
                    if not (ANSWER_PREFIX.format(label) in answer):
                        # Correct case; we'll
                        success_attacks += 1
                        count_paras_success += 1
                        map_paras_total_success[len(p_info_list)] += 1

                        map_total_success_by_subject[info['subject']] += 1

            attack_success_counts.append(success_attacks)
            if success_attacks > 0:
                map_paras_success[len(p_info_list)] += 1

                map_1success_by_subject[info['subject']] += 1

    acc = correct_count / len(q_info_list)
    print("Stats for first pass: ")
    print("Count correct: " + str(correct_count))
    print("Accuracy: " + str(acc))
    print("Count noncompliant: " + str(noncomply_count))
    print("Total number of questions parsed: " + str(len(q_info_list)))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # how many had any attack success?
    count_attacks = 0
    for successes in attack_success_counts:
        if successes > 0:
            count_attacks += 1

    # Filter questions with no successful paraphrases?

    if count_actually_para == 0:
        return

    print("Questions with conflicting duplicate paraphrases: " + str(conflicting_output_count))
    print("Number of questions with one successful paraphrase: " + str(count_actually_para))
    print("Proportion of questions with one successful paraphrase: " + str(count_actually_para / len(q_info_list)))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Number of questions with one successful attack: " + str(count_attacks))
    print("Proportion of questions with one successful attack: " + str(count_attacks/len(q_info_list)))



    print("ASR, with questions with at least one paraphrase: " + str(count_attacks / count_actually_para))

    print("Total Num Paraphrases: " + str(count_paras))
    print("Attack Successes/Num Paraphrases: " + str(count_paras_success / count_paras))

    print("Attack stats per Question: ")
    print("Min success/Question: " + str(np.min(attack_success_counts)))
    print("Mean success/Question: " + str(np.mean(attack_success_counts)))
    print("Max success/Question: " + str(np.max(attack_success_counts)))

    # Condition on type of question we paraphrase?
    # plt.hist(count_paras_dist, bins=list(range(0,21)))
    paras_flat = []
    asr_per_nums = []
    total_asr_per_nums = []
    for b in range(0, 21):
        sub_asr = map_paras_success[b] / map_paras_freq[b]
        asr_per_nums.append(sub_asr)
        paras_flat.append(map_paras_freq[b])
        if map_paras_total_count[b] != 0:
            total_asr_per_nums.append(map_paras_total_success[b] / map_paras_total_count[b])
        else:
            total_asr_per_nums.append(0)

    plt.bar(x=list(range(0, 21)), height=paras_flat)
    plt.title("Distribution of Paraphrases per Question")
    plt.xlabel("Number of Paraphrases")
    plt.ylabel("Count")
    plt.show()


    attack_success_counts_plt = [0] * 21
    for item in attack_success_counts:
        attack_success_counts_plt[item] += 1

    # Skip zero for the sake of plotting (We know many are unsuccessful already!)
    plt.bar(x=list(range(1, 21)), height=attack_success_counts_plt[1:])
    plt.title("Number Successes, vs. Number of Paraphrases Per Question")
    plt.xlabel("Number of Paraphrases")
    plt.ylabel("Number of Successes")
    plt.show()

    # Sanity plot: ASR vs number of paraphrases.
    plt.bar(x=list(range(0,21)), height=asr_per_nums)
    plt.title("1-Success Rate vs. Number of Paraphrases Per Question")
    plt.xlabel("Number of Paraphrases")
    plt.ylabel("Success Rate")
    plt.show()

    plt.bar(x=list(range(0,21)), height=total_asr_per_nums)
    plt.title("Total Success Rate vs. Number of Paraphrases Per Question")
    plt.xlabel("Number of Paraphrases")
    plt.ylabel("Success Rate")
    plt.show()

    bar_subjs = []
    asr1s = []
    asr_totals = []
    count_totals = []
    avg_paras_subject = []
    for c in map_qs_by_subject.keys():
        bar_subjs.append(c)
        asr1s.append(map_1success_by_subject[c]/map_qs_by_subject[c])
        asr_totals.append(map_total_success_by_subject[c]/map_paras_by_subject[c])
        count_totals.append(map_qs_by_subject[c])
        avg_paras_subject.append(map_paras_by_subject[c]/map_qs_by_subject[c])

    # Trying to get the labels to be more readable
    # https://stackoverflow.com/questions/10998621/rotate-axis-tick-labels

    # Interesting note: No correlation with question count.
    # argsorted_subject_count = range(len(bar_subjs))
    argsorted_subject_count = sorted(range(len(bar_subjs)), key=lambda x: count_totals[x])
    # No correlation with average number of paraphrases created per subject.
    # argsorted_subject_count = sorted(range(len(bar_subjs)), key=lambda x: avg_paras_subject[x])


    bar_subjs = [bar_subjs[x] for x in argsorted_subject_count]
    count_totals = [count_totals[x] for x in argsorted_subject_count]
    asr1s = [asr1s[x] for x in argsorted_subject_count]
    asr_totals = [asr_totals[x] for x in argsorted_subject_count]
    avg_paras_subject = [avg_paras_subject[x] for x in argsorted_subject_count]

    # Some string processing
    max_label_chars = 9
    bar_subjs = [x.replace('_', ' ')[:max_label_chars] + "."  for x in bar_subjs]

    # number of questions by subject
    plt.bar(x=range(len(bar_subjs)), height=count_totals, tick_label=bar_subjs)
    # plt.bar(x=range(len(bar_subjs)), height=count_totals)
    plt.title("Number of Correctly Answered Questions per Subject")
    plt.xlabel("Subject")
    plt.ylabel("Num Qs")
    plt.xticks(rotation=45, ha='right') # fontsize=6
    # plt.legend(loc='best', fontsize=8)
    plt.show()

    plt.bar(x=range(len(bar_subjs)), height=avg_paras_subject, tick_label=bar_subjs)
    plt.title("Avg. Number of Paraphrases Per Question by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Avg. Paraphrases")
    plt.xticks(rotation=45, ha='right') # fontsize=6
    # plt.legend(loc='best', fontsize=8)
    plt.show()

    # 1asr by subject
    plt.bar(x=range(len(bar_subjs)), height=asr1s, tick_label=bar_subjs)
    plt.title("Question 1-Success Rate by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45, ha='right') # fontsize=6
    # plt.legend(loc='best', fontsize=8)
    plt.show()

    # overall asr by subject
    plt.bar(x=range(len(bar_subjs)), height=asr_totals, tick_label=bar_subjs)
    plt.title("Total Success Rate by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Outliers here
    # High school math -
    # Abstract Algebra -
    # Formal Logic -
    # Moral Scenarios

def screen_complex_questions(question_info):
    q_str = question_info['questio_only']
    if '\n' in q_str: # multiline not well supported
        return False
    elif '__' in q_str: # Substitution questions don't paraphrase well.
        return False
    else:
        return True


def collect_sampling_of_para_qs():
    pass

if __name__ == "__main__":
    # make_request("questions_dev.pickle", 'dev')
    # Sent batch...
    # Batch id: batch_67ed6e4231988190a552d62fb967ed70
    # Fid: file-7DWDJxJdxhk7TkaoSadRTT
    # process_batch("questions_dev.pickle", 'openai_mmlu_dev.jsonl', "questions_dev_answered.pickle")
    # check_stats("questions_dev_answered.pickle")

    # request_para_answers('para_dev_eval_0.pickle', 'para_dev_eval_final.pickle')
    # Sent batch...
    # Batch id: batch_67f54536183081909785d6fb8b572d27
    # Fid: file-NC8tQXaiagzRPn8oYr2SMo

    # Full test set request.
    # make_request("questions_test.pickle", 'test')
    # Sent batch...
    # Batch id: batch_67ed7cf9dcb081909c76b814722e511b
    # Fid: file-JrZfrmUvtfddmLavZuTGju
    # process_batch("questions_test.pickle", 'openai_mmlu_test.jsonl', "questions_test_answered.pickle")
    # check_stats("questions_test_answered.pickle")

    # request_para_answers('para_batches_evaled.pickle', 'para_test_eval_final.pickle')
    # request_para_answers('para_test_eval_final.pickle', 'para_test_eval_final.pickle')
    # Building request to evaluate paraphrases...
    # Making sub requests...
    # Sub batch: 0
    # Sub batch: Number of requests: 50000
    # Batch id: batch_67f549b2b65c819098f4bd0eb8a2d57f
    # Fid: file-BwUvAicRnJRNcTCaPruwMp
    # Sub batch: 1
    # Sub batch: Number of requests: 50000
    # Batch id: batch_67f54a05c468819099f9aebc5e5cd17c
    # Fid: file-9XqiHNRF57i19mScpBfMjW
    # Sub batch: 2
    # Sub batch: Number of requests: 29554
    # Batch id: batch_67f54a4597408190bfda3644fec7d07a
    # Fid: file-3T6NRHqUDoV8i6251MKRVt

    # Last batch again
    # Sub batch: 2
    # Sub batch: Number of requests: 29554
    # Batch id: batch_67f5d125515c8190b22e5bbb9be7a92f
    # Fid: file-TqyahyJNnp2EfdC2FpBfyz

    # Dev.
    # process_batch('para_dev_eval_final.pickle', 'dev_para_responses.jsonl', 'para_dev_eval_results.pickle')

    # All of the test.
    # process_batch('para_test_eval_final.pickle', 'test_para_responses_0.jsonl', 'para_test_eval_results.pickle') # Doesn't link?
    # process_batch('para_test_eval_results.pickle', 'test_para_responses_1.jsonl', 'para_test_eval_results.pickle') # Doesn't link?
    # process_batch('para_test_eval_results.pickle', 'test_para_responses_2.jsonl', 'para_test_eval_results.pickle')

    # compare_uuid_sets('para_test_eval_final.pickle','test_para_responses_0.jsonl')
    # compare_uuid_sets('para_test_eval_final.pickle', 'test_para_responses_1.jsonl')
    # compare_uuid_sets('para_test_eval_final.pickle', 'test_para_responses_2.jsonl')

    # Rerun first 2 batches due to a uuid error:
    # Making sub requests...
    # Sub batch: 0
    # Sub batch: Number of requests: 50000
    # Batch id: batch_67f6a13b783081908ceffa72c02d72d0
    # Fid: file-Vvu5dn9SMSCESPyDydkdq1
    # Sub batch: 1
    # Sub batch: Number of requests: 50000
    # Batch id: batch_67f6a18aa50081909dea692fb926f8aa
    # Fid: file-HYe1X8y9ExBRH6bqJuxvqv

    # print("Stats, No question condition.~~~~~~~~~~~~~~~~~~~~~")
    # check_stats('para_dev_eval_results.pickle')
    #
    # print("Stats, Likely paraphrase rules condition.~~~~~~~~~")
    # # Paraphrasing isn't perfect; we'll rule out some:
    # # Multiline questions
    # # Underscore/substitution questions.
    # check_stats('para_dev_eval_results.pickle', question_condition=screen_complex_questions)

    # print("Test Stats, No question condition.~~~~~~~~~~~~~~~~~~~~~")
    # check_stats('para_test_eval_results.pickle')

    print("Test Stats, Likely paraphrase rules condition.~~~~~~~~~")
    # Paraphrasing isn't perfect; we'll rule out some:
    # Multiline questions
    # Underscore/substitution questions.
    check_stats('para_test_eval_results.pickle', question_condition=screen_complex_questions)


    print("Done")

    # Dev results

    # Stats, No question condition.~~~~~~~~~~~~~~~~~~~~~
    # Stats for first pass:
    # Accuracy: 1.0
    # Count noncompliant: 0
    # Total number of questions parsed: 213
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Questions with conflicting duplicate paraphrases: 2
    # Number of questions with one successful paraphrase: 213
    # Proportion of questions with one successful paraphrase: 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Number of questions with one successful attack: 75
    # Proportion of questions with one successful attack: 0.352112676056338
    # ASR, with questions with at least one paraphrase: 0.352112676056338
    # Attack Successes/Num Paraphrases: 0.15191919191919193
    # Attack stats per Question:
    # Min success/Question: 0
    # Mean success/Question: 1.7652582159624413
    # Max success/Question: 18
    # Stats, Likely paraphrase rules condition.~~~~~~~~~
    # Stats for first pass:
    # Accuracy: 1.0
    # Count noncompliant: 0
    # Total number of questions parsed: 181
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Questions with conflicting duplicate paraphrases: 2
    # Number of questions with one successful paraphrase: 181
    # Proportion of questions with one successful paraphrase: 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Number of questions with one successful attack: 65
    # Proportion of questions with one successful attack: 0.35911602209944754
    # ASR, with questions with at least one paraphrase: 0.35911602209944754
    # Attack Successes/Num Paraphrases: 0.14968999114260406
    # Attack stats per Question:
    # Min success/Question: 0
    # Mean success/Question: 1.867403314917127
    # Max success/Question: 18
    # Done

    # Test results

    # Test Stats, No question condition.~~~~~~~~~~~~~~~~~~~~~
    # Stats for first pass:
    # Accuracy: 1.0
    # Count noncompliant: 0
    # Total number of questions parsed: 10521
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Questions with conflicting duplicate paraphrases: 108
    # Number of questions with one successful paraphrase: 10521
    # Proportion of questions with one successful paraphrase: 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Number of questions with one successful attack: 3708
    # Proportion of questions with one successful attack: 0.35243798118049613
    # ASR, with questions with at least one paraphrase: 0.35243798118049613
    # Attack Successes/Num Paraphrases: 0.13381260836850198
    # Attack stats per Question:
    # Min success/Question: 0
    # Mean success/Question: 1.5550803155593576
    # Max success/Question: 20
    # Test Stats, Likely paraphrase rules condition.~~~~~~~~~
    # Stats for first pass:
    # Accuracy: 1.0
    # Count noncompliant: 0
    # Total number of questions parsed: 9262
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Questions with conflicting duplicate paraphrases: 99
    # Number of questions with one successful paraphrase: 9262
    # Proportion of questions with one successful paraphrase: 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Number of questions with one successful attack: 3313
    # Proportion of questions with one successful attack: 0.3576981213560786
    # ASR, with questions with at least one paraphrase: 0.3576981213560786
    # Attack Successes/Num Paraphrases: 0.13075427993290556
    # Attack stats per Question:
    # Min success/Question: 0
    # Mean success/Question: 1.5906931548261714
    # Max success/Question: 20
    # Done