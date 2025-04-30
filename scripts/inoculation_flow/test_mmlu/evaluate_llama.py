import argparse
import openai
import os
import numpy as np
import pandas as pd
import time

from crop import crop

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

# Explanation of cached models:
# https://github.com/huggingface/diffusers/issues/1440

# from transformers import AutoTokenizer
# import transformers
#
# access_token = "Enter your token here"
# model = "meta-llama/Llama-2-7b-chat-hf"
#
# tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
#
# model = AutoModelForCausalLM.from_pretrained(
#     model,
#     token=access_token
# )
#
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     tokenizer=tokenizer,
#     device_map="auto",
# )

choices = ["A", "B", "C", "D"]

# This does work, but the download is huge
# test with tinyllama to debug code...

# non-rlhf (alignment) version
# TEST_MODEL = "Llama-2-7b-hf"
#
# MODEL_REPO = "meta-llama/Llama-2-7b-hf"
# EXPECTED ACCURACY: 45.3, from llama paper: https://arxiv.org/pdf/2307.09288

TEST_MODEL = "TinyLlama-1.1B-Chat-v1.0"

MODEL_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TinyLlama seems to be acting weird. OpenLLama?
# TEST_MODEL = "open_llama_3b_v2"
#
# MODEL_REPO = "openlm-research/open_llama_3b_v2"

# Too big! immediate sigkill...



# Different TinyLLama version?
TEST_MODEL = "TinyLlama_v1.1"

MODEL_REPO = "TinyLlama/TinyLlama_v1.1"

TOP_K = 100  # We look these K highest scores.

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt



def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

# Modified form for instruct prompting for llama
def format_example_no_answ(df, idx):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    return prompt

# Describing llama's format from their eval data:
# The following is for the 5 shot instruct model.
# \n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {question}
# \nYour response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.
# The 5-shot mode has a lot of context text in it...

# each example is formatted in llama's instruction mode
# But the whole question (examples + real q) is surrounded by <|begin_of_text|> {stuff} <|end_of_text|>

def gen_prompt_instruct(question, dev_df, tokenizer, k=-1):

    if k == -1:
        k = dev_df.shape[0]

    prompt_turn_string = ("\n\nGiven the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {}\n"
                          "Your response should end with \"The best answer is [the_answer_letter]\" where the [the_answer_letter] is one of A, B, C or D.")

    response_template = "\n\nThe best answer is {}."
    response_final = "\n\nThe best answer is"

    msgs = []

    for i in range(k):
        # Build messages
        msgs.append({"role": "user", "content": prompt_turn_string.format(format_example_no_answ(dev_df, i))})
        # Get the answer.
        ans = dev_df.iloc[i, dev_df.shape[1]-1]  # 0 index???

        msgs.append({"role": "assistant", "content": response_template.format(ans)})


    # append the real question.
    msgs.append({"role": "user", "content": prompt_turn_string.format(question)})
    msgs.append({"role": "assistant", "content": response_final})

    # Return the completed template.
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def eval(args, subject, pipeline, model, tokenizer, dev_df, test_df):

    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        print("Q num: " + str(i) + "/" + str(test_df.shape[0]))
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example_no_answ(test_df, i)
        # train_prompt = gen_prompt(dev_df, subject, k)
        prompt = gen_prompt_instruct(prompt_end, dev_df, pipeline.tokenizer, k)
        print(prompt)

        # Weird encoding stuff going on here.
        # prompt = train_prompt + prompt_end

        # while crop(prompt) != prompt:
        #     k -= 1
        #     train_prompt = gen_prompt(dev_df, subject, k)
        #     prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        # while True:
        #     try:
                # break
            # except Exception as e: # JL Note: Remove retry loop sometime; LLaMA is running locally...
            #     print("Exception: " + str(e))
            #     print("pausing")
            #     time.sleep(1)
            #     continue


        msgs = [{"role": "system", "content": train_prompt},
                      {"role": "user", "content": prompt_end}]

        print("Message: ")
        print(msgs)

        prompt = pipeline.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Note: it doesn't appear that openllama was trained using the same prompt template? Getting terrible results!
        # c = pipeline(prompt, max_new_tokens=1, do_sample=False)

        encoded = tokenizer(prompt, return_tensors="pt")
        generated_output = model.generate(**encoded, do_sample=True, temperature=1.0, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, output_logits=True)

        # generated_tokens = model.generate(**encoded, do_sample=True, temperature=1.0, max_new_tokens=1)

        token_list = [tokenizer.decode(x) for x in generated_output.sequences[0]]

        logit_set = generated_output.logits

        # make numpy array
        logit_set = logit_set[0].cpu().detach().numpy()

        # get top 100
        k = logit_set.shape[1] - TOP_K
        top_tokens = np.argpartition(logit_set, k, axis=1)[0][-TOP_K::] # [-TOP_K::]

        top_logprobs = logit_set[0][top_tokens]

        # Decode
        top_tokens = [tokenizer.convert_ids_to_tokens([x])[0] for x in top_tokens]

        # Interesting subtlety, which "appears" to create duplicates:
        # https://discuss.huggingface.co/t/llama2-tokenizer-duplicate-ids/51708
        # I used the convert_ids_to_tokens() method which generates the underscore for those particular tokens.

        lprobs = []

        for ans in answers:
            try:
                # lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
                prob_loc = top_tokens.index(ans)
                lprobs.append(top_logprobs[prob_loc])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):
    # engines = args.engine

    engines = [TEST_MODEL]# use model name instead

    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine)))

    print(subjects)
    print(args)

    # JL - engine is just the model name now.
    for engine in engines:

        all_cors = []

        # Initialize model, tokenizer, pipeline
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     MODEL_REPO,
        #     token=API_KEY
        # )
        # tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO, token=API_KEY)

        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_REPO)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_REPO)

        # Reference link: https://discuss.huggingface.co/t/is-transformers-using-gpu-by-default/8500

        # checking gpu status
        if torch.cuda.is_available():
            print("GPU available...")
        else:
            print("No GPU available...")

        # model = LlamaForCausalLM.from_pretrained(MODEL_REPO)
        # tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_REPO)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,  # Probably won't work on cpu...
            tokenizer=tokenizer,
            device_map="auto",
        )
        print(engine)

        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs = eval(args, subject, pipeline, model, tokenizer, dev_df, test_df)
            all_cors.append(cors)

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada"],
    #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
    args = parser.parse_args()
    main(args)

