# Manage evaluating paraphrase quality
# jlim@wpi.edu

import sys, os
import pickle
import re
import random
import copy

import pandas as pd
import numpy as np
import scipy

ANSWER_PREFIX = "The best answer is {}"

ANSWER_REG = "The best answer is [ABCD]"

def screen_complex_questions(question_info):
    q_str = question_info['questio_only']
    if '\n' in q_str: # multiline not well supported
        return False
    elif '__' in q_str: # Substitution questions don't paraphrase well.
        return False
    else:
        return True


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

def build_df_sampling_weighted(in_pickle, out_pickle, sample_num=2000, screen_category_list=None, do_success_only=False):
    with open(in_pickle, 'rb') as f:
        question_set = pickle.load(f)

    # Add a condition; for the question/answers.
    question_set = list(filter(screen_complex_questions, question_set))

    print("Question set size: " + str(len(question_set)))

    print("Adjusting for paraphrase difficulty, subject.")
    # Build conditional weighting.
    # Control for subject and paraphrase difficulty
    weighting_map = {}
    question_choice_list = []
    question_paracount_list = []
    for q_info in question_set:

        if screen_category_list is not None:
            if q_info['subject'] not in screen_category_list:
                continue

        if 'evaled_qinfos' in q_info:
            p_info_list, conflict = deduplicate_para_qs(q_info['evaled_qinfos'])
        else:
            continue

        if not conflict and len(p_info_list) != 0:
            if do_success_only:
                # Only allow successful questions.
                label = q_info['answer']
                success = False
                for p_info in p_info_list:
                    answer = p_info['model_response']
                    # We ignore noncompliant responses.
                    if re.search(ANSWER_REG, answer) is not None:
                        # Does the answer match? Yes or no...
                        if not (ANSWER_PREFIX.format(label) in answer):
                            success = True

                # In this case, skip this particular question.
                if not success:
                    continue

            question_choice_list.append(q_info)
            question_paracount_list.append(len(p_info_list))
            if q_info['subject'] not in weighting_map:
                weighting_map[q_info['subject']] = np.zeros(21)

            # Accumulate count.
            weighting_map[q_info['subject']][len(p_info_list)] += 1

    # total_q_num = len(question_choice_list)

    # Screen the following keys:
    #
    #
    # if screen_category_list is not None:
    #     # Remove the other categories not in this list.
    #     keylist = list(weighting_map.keys())
    #     for key in keylist:
    #         if key not in screen_category_list:
    #             del weighting_map[key]

    print("Normalize weights.")
    # num_categories = len(list(weighting_map.keys())) * 20

    # Need to count nonzero to get the number of categories.
    num_categories = 0
    flatten_counts = []
    for key in weighting_map.keys():
        num_categories += np.count_nonzero(weighting_map[key])

        nonzeroed = []
        for item in weighting_map[key]:
            if item != 0:
                nonzeroed.append(item)
        flatten_counts = flatten_counts + nonzeroed

    # sanity check.
    test_flatnorm = 1.0 / (np.array(flatten_counts) * num_categories)
    flatnorm_sum = np.sum(test_flatnorm * np.array(flatten_counts))

    print("Number of categories: " + str(num_categories))
    # partial sum for checking
    weight_sum = 0
    for key in weighting_map.keys():
        weighting_map[key] = 1.0 / (weighting_map[key] * num_categories)
        # Zero out infinites - not selected!
        weighting_map[key][weighting_map[key] == np.inf] = 0
        partial_sum = np.sum(weighting_map[key])
        print("Partial sum: " + str(partial_sum))
        weight_sum += partial_sum

    # partial_sum = 0
    # for key in weighting_map.keys():
    #     partial_sum += np.sum(weighting_map[key])

    print("Weighting map sum: " + str(weight_sum))
    # print("Size of question choice list: " + str(len(question_choice_list)))

    # build weightlist.
    question_choice_weights = []
    for idx, q in enumerate(question_choice_list):
        weight_value = weighting_map[q['subject']][question_paracount_list[idx]]
        # weight_value = weight_value / total_q_num
        # Odds across subject category, paraphrase difficulty balanced.

        question_choice_weights.append(weight_value)

    if sample_num > len(question_choice_list):
        # argchoice
        questions_sampled = np.random.choice(len(question_choice_list), size=len(question_choice_list), replace=False, p=question_choice_weights)
    else:
        # argchoice
        questions_sampled = np.random.choice(len(question_choice_list), size=sample_num, replace=False, p=question_choice_weights)


    output = {
        'Original_Question': [],
        'Paraphrase_Question': [],
        'Answer_Choices': [],
        'Is_Paraphrase': []
    }
    # output question + answers, random para + answers to pandas dataframe.
    for argsample in questions_sampled:
        question_choice = question_choice_list[argsample]

        orig_str = question_choice['question']
        orig_question = question_choice['questio_only']

        #                 'questio_only': data_df.iloc[q_idx, 0],
        #                 'question': format_question_str,
        question_suffix = question_choice['question'][len(question_choice['questio_only']):]

        para_q = None
        if do_success_only:
            shuffled_set = copy.deepcopy(question_choice['evaled_qinfos'])
            random.shuffle(shuffled_set)
            label = question_choice['answer']
            for p_info in shuffled_set:
                answer = p_info['model_response']
                # We ignore noncompliant responses.
                if re.search(ANSWER_REG, answer) is not None:
                    # Does the answer match? Yes or no...
                    if not (ANSWER_PREFIX.format(label) in answer):
                        para_q = p_info
        else:
            para_q = random.choice(question_choice['evaled_qinfos'])

        if para_q is None:
            print("Logic issue!")

        output['Original_Question'].append(orig_question)
        output['Paraphrase_Question'].append(para_q['para_q'])
        output['Answer_Choices'].append(question_suffix)
        output['Is_Paraphrase'].append("Unknown")

        # para_str = para_q['para_q'] + question_suffix

        # print("Testing")
        # print("Original: ")
        # print(orig_str)
        # print("Paraphrase: ")
        # print(para_str)

        # return

    print("Writing " + str(len(output['Original_Question'])) + " records...")
    with open(out_pickle, 'wb') as f:
        pickle.dump(output, f)

    # # Add to df.
    # # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    # df_output = pd.DataFrame(data=output)
    #
    # print("Writing " + str(df_output.shape[0]) + " records...")
    # # Save df
    # df_output.to_csv(out_df)


# Got this error: /home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/Attacks/Manual_Para_Eval.py:219: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame
# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   examples['Is_Paraphrase'][para_idx] = user_answer

def clear_term():
    print("\033[H\033[J", end="")

    # 50 newlines - hacky.
    # print("\n"*50)


def make_slicedict(dictionary, last_idx):
    newdict = {}
    for key in dictionary.keys():
        newdict[key] = dictionary[key][:last_idx]

    return newdict

def eval_paraphrases(pickle_path, out_df_path):
    # code to evaluate a set of paraphrases manually.

    # examples = pd.read_csv(df_path)
    #
    # # Convert to dictionary
    # examples_dict = examples.to_dict()
    with open(pickle_path, 'rb') as f:
        examples_dict = pickle.load(f)

    num_examples = len(examples_dict['Original_Question'])

    # Find first question without 'yes' or 'no' response
    for para_idx in range(num_examples):
        answer_str = examples_dict['Is_Paraphrase'][para_idx]
        if (answer_str != 'yes' and answer_str != 'no'):
            print("Questions Evaluated/Total: " + str(para_idx) + '/' + str(num_examples))
            # Give to the user; pass otherwise.
            print("Original Question:")
            print(examples_dict['Original_Question'][para_idx])
            print("Paraphrase question:")
            print(examples_dict['Paraphrase_Question'][para_idx])
            print("Answer choices:")
            print(examples_dict['Answer_Choices'][para_idx])

            user_answer = 'asdfa'
            while user_answer != 'y' and user_answer != 'n' and user_answer != 'q':
                user_answer = input("Is the question paraphrased (y/n, q exit)?: ")

            if user_answer == 'y':
                user_answer = 'yes'
            elif user_answer == 'n':
                user_answer = 'no'
            else:
                # Quit
                return

            examples_dict['Is_Paraphrase'][para_idx] = user_answer
            # examples.loc['Is_Paraphrase', para_idx] = user_answer
            # Write to csv file for now.

            # not efficient, but whatever.
            df_output = pd.DataFrame(data=make_slicedict(examples_dict, para_idx+1))

            # Save df
            df_output.to_csv(out_df_path)

            # Update pickle.
            with open(pickle_path, 'wb') as f2:
                pickle.dump(examples_dict, f2)

            # Clear terminal.
            # Useful tips/codes for clearing the console between evals:
            # https://stackoverflow.com/questions/517970/how-can-i-clear-the-interpreter-console
            # https://en.wikipedia.org/wiki/ANSI_escape_code
            # print("\033[H\033[J", end="")
            clear_term()

def view_paraphrases(pickle_path):
    with open(pickle_path, 'rb') as f:
        examples_dict = pickle.load(f)

    num_examples = len(examples_dict['Original_Question'])

    # Find first question without 'yes' or 'no' response
    for para_idx in range(num_examples):
        answer_str = examples_dict['Is_Paraphrase'][para_idx]
        if ( answer_str == 'no'):
            print("Questions Evaluated/Total: " + str(para_idx) + '/' + str(num_examples))
            # Give to the user; pass otherwise.
            print("Original Question:")
            print(examples_dict['Original_Question'][para_idx])
            print("Paraphrase question:")
            print(examples_dict['Paraphrase_Question'][para_idx])
            print("Answer choices:")
            print(examples_dict['Answer_Choices'][para_idx])
            print("Is Paraphrase:")
            print(examples_dict['Is_Paraphrase'][para_idx])

            input("")

            # Clear terminal.
            # Useful tips/codes for clearing the console between evals:
            # https://stackoverflow.com/questions/517970/how-can-i-clear-the-interpreter-console
            # https://en.wikipedia.org/wiki/ANSI_escape_code
            # print("\033[H\033[J", end="")
            # clear_term()


def plot_eval(pickle_path):

    print("Results for file: " + str(pickle_path))
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)

    count_yes = 0
    count_no = 0

    for idx in range(len(results['Original_Question'])):
        response = results['Is_Paraphrase'][idx]
        if response == 'yes':
            count_yes += 1
        elif response == 'no':
            count_no += 1

    total_answered = count_yes + count_no
    print("Total answered: " + str(total_answered))
    print("Count Yes: " + str(count_yes))
    print("Count No: " + str(count_no))
    print("Fraction Yes: " + str(count_yes/total_answered))


def get_ci(num_trials, proportion, confidence=0.9):
    # Assuming binomial.

    interval = scipy.stats.binom.interval(n=num_trials, p=proportion, confidence=confidence)
    print("Proportion: " + str(proportion) + "; Interval: " + str(interval[0]/num_trials) + "-" + str(interval[1]/num_trials))

def main():
    # Build set to evaluate
    results_pickle = '/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/Attacks/para_test_eval_results.pickle'
    # Evaluate all categories
    # build_df_sampling_weighted(results_pickle, 'eval_all.pickle', sample_num=2000)

    # eval_paraphrases('eval_all.pickle', 'eval_all.csv')

    view_paraphrases('eval_all_correct.pickle')

    # Evaluate just abstract algebra, moral dilemmas
    # build_df_sampling_weighted(results_pickle, 'eval_moral.pickle', sample_num=2000,
    #                            screen_category_list=['moral_scenarios'])
    #
    # eval_paraphrases('eval_moral.pickle', 'eval_moral.csv')
    #
    # build_df_sampling_weighted(results_pickle, 'eval_alg.pickle', sample_num=2000, screen_category_list=['abstract_algebra'])
    #
    # eval_paraphrases('eval_alg.pickle', 'eval_alg.csv')

    # Evaluate successful answers.
    # build_df_sampling_weighted(results_pickle, 'eval_all_correct.pickle', sample_num=2000, screen_category_list=None,
    #                            do_success_only=True)

    # eval_paraphrases('eval_all_correct.pickle', 'eval_all_correct.csv')

    # plot_eval('eval_all.pickle')
    # plot_eval('eval_alg.pickle')
    # plot_eval('eval_moral.pickle')

    # plot_eval('eval_all_correct.pickle')

    print("Displaying 90% confidence intervals: ")
    print("Manual eval, attack success: ")
    get_ci(101, 0.2673, confidence=0.9)
    print("Manual eval, all subjects: ")
    get_ci(200, 0.605, confidence=0.9)
    print("Manual eval, alg: ")
    get_ci(33, 0.1515, confidence=0.9)
    print("Manual eval, moral: ")
    get_ci(51, 0.19607, confidence=0.9)

    print("Done")



if __name__ == "__main__":
    main()


# RESULTS:
# Results for file: eval_all.pickle
# Total answered: 200
# Count Yes: 121
# Count No: 79
# Fraction Yes: 0.605
# Results for file: eval_alg.pickle
# Total answered: 33
# Count Yes: 5
# Count No: 28
# Fraction Yes: 0.15151515151515152
# Results for file: eval_moral.pickle
# Total answered: 51
# Count Yes: 10
# Count No: 41
# Fraction Yes: 0.19607843137254902
# Done

# For para success only:
# Results for file: eval_all_correct.pickle
# Total answered: 101
# Count Yes: 27
# Count No: 74
# Fraction Yes: 0.26732673267326734

# Manual eval, attack success:
# Proportion: 0.2673; Interval: 0.19801980198019803-0.33663366336633666
# Manual eval, all subjects:
# Proportion: 0.605; Interval: 0.55-0.66
# Manual eval, alg:
# Proportion: 0.1515; Interval: 0.06060606060606061-0.2727272727272727
# Manual eval, moral:
# Proportion: 0.19607; Interval: 0.11764705882352941-0.29411764705882354
# Done