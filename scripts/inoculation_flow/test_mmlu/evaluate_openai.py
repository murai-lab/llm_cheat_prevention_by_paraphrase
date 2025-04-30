import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
import json
import io

from crop import crop

# Modified for testing by Jeremy Lim
# jlim@wpi.edu

KEYPATH = ""
with open(KEYPATH, 'r') as f:
    API_KEY = f.readline()
    API_KEY = API_KEY.rstrip('\n')

choices = ["A", "B", "C", "D"]

TEST_MODEL = "gpt-3.5-turbo-0125"

DEBUGGING = False

BATCH_CHUNK_SIZE = 250  # Largest # of questions we can send in one batch, to avoid token limits.
 # 250
# Pricing
# Input: $0.50 per 1 million token
# Output: $1.50 per 1 million token

COST_TEST_MODE = False  # Run the code in cost test mode; try to estimate total cost of the operation.

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
        time.sleep(3)
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
        time.sleep(3)  # Wait period, so as to not spam polling...
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

    # list of top logprobs: x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    top_tokens_lists = [[y["token"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]
    top_logprobs_lists = [[y["logprob"] for y in x["response"]["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]] for x in responses]

    print("done")
    files_list = openai_client.files.list()
    print("Files remaining on openAI api: " + str(len(files_list.data)))

    return top_tokens_lists, top_logprobs_lists

def eval(args, subject, model_str, dev_df, test_df):

    client = openai.OpenAI(api_key=API_KEY)

    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    in_token_count = 0
    out_token_count = 0

    batch_param_sets = []

    # create one large batch, then process results in a later loop.
    for i in range(test_df.shape[0]):
        print("Batching Q num: " + str(i+1) + "/" + str(test_df.shape[0]))
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        # time.sleep(0.15)  # Ensure below 500 Requests Per Minute
        # TODO: Implement batching...

        msgs = [{"role": "system", "content": train_prompt},
                      {"role": "user", "content": prompt_end}]

        param_set = {
            'model': model_str,
            'messages': msgs,
            'logprobs': True,
            'max_tokens': 1,
            'temperature': 0,
            'top_logprobs': 20
        }

        batch_param_sets.append(param_set)

    # Evaluate batches in chunks.
    top_token_lists = []
    top_logprobs_lists = []

    num_completions = len(batch_param_sets)
    num_sub_batches = int(num_completions/BATCH_CHUNK_SIZE)
    remainder = num_completions % BATCH_CHUNK_SIZE
    for i in range(num_sub_batches):
        toks, lpro = batch_eval(batch_param_sets[i*BATCH_CHUNK_SIZE:(i+1)*BATCH_CHUNK_SIZE], client)

        top_token_lists = top_token_lists + toks
        top_logprobs_lists = top_logprobs_lists + lpro

    if remainder != 0: # Handle remainder.
        toks, lpro = batch_eval(batch_param_sets[-remainder:], client)

        top_token_lists = top_token_lists + toks
        top_logprobs_lists = top_logprobs_lists + lpro

    for i in range(len(top_token_lists)):

        top_tokens = top_token_lists[i]
        # the top 20 tokens
        top_logprobs = top_logprobs_lists[i]
        # the top 20 associated logprobs.

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

    # for i in range(test_df.shape[0]):
    #     print("Q num: " + str(i) + "/" + str(test_df.shape[0]))
    #     # get prompt and make sure it fits
    #     k = args.ntrain
    #     prompt_end = format_example(test_df, i, include_answer=False)
    #     train_prompt = gen_prompt(dev_df, subject, k)
    #     prompt = train_prompt + prompt_end
    #
    #     while crop(prompt) != prompt:
    #         k -= 1
    #         train_prompt = gen_prompt(dev_df, subject, k)
    #         prompt = train_prompt + prompt_end
    #
    #     label = test_df.iloc[i, test_df.shape[1]-1]
    #
    #     # time.sleep(0.15)  # Ensure below 500 Requests Per Minute
    #     # TODO: Implement batching...
    #
    #     msgs = [{"role": "system", "content": train_prompt},
    #                   {"role": "user", "content": prompt_end}]
    #
    #     print("Message: ")
    #     print(msgs)
    #
    #     param_set = {
    #         'model': model_str,
    #         'messages': msgs,
    #         'logprobs': True,
    #         'max_tokens': 1,
    #         'temperature': 0,
    #         'top_logprobs': 20
    #     }
    #
    #     batch_eval([param_set, param_set], client)
    #
    #     if not COST_TEST_MODE:
    #         c = client.chat.completions.create(
    #             model=model_str,
    #             messages=msgs,
    #             logprobs=True,
    #             max_tokens=1,
    #             temperature=0,
    #             top_logprobs=20,
    #         )
    #
    #     # system, user, assistant: Which parts of the prompt go where?
    #     # https://community.openai.com/t/how-to-design-few-shot-prompt-with-api/656727
    #     # From this example, put it in the system message
    #
    #     # c = openai.Completion.create(
    #     #     engine=engine,
    #     #     prompt=prompt,
    #     #     max_tokens=1,
    #     #     logprobs=100,
    #     #     temperature=0,
    #     #     echo=True
    #     # )
    #
    #
    #     # Token counts, use for cost estimation
    #
    #     for msg in msgs:
    #         in_token_count += len(msg["content"])
    #
    #     out_token_count += 1
    #
    #     lprobs = []
    #
    #     if not COST_TEST_MODE:
    #
    #         top_tokens = [x.token for x in c.choices[0].logprobs.content[0].top_logprobs]
    #         # the top 20 tokens
    #         top_logprobs = [x.logprob for x in c.choices[0].logprobs.content[0].top_logprobs]
    #         # the top 20 associated logprobs.
    #
    #         for ans in answers:
    #             try:
    #                 # lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
    #                 prob_loc = top_tokens.index(ans)
    #                 lprobs.append(top_logprobs[prob_loc])
    #             except:
    #                 print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
    #                 lprobs.append(-100)
    #         pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
    #         probs = softmax(np.array(lprobs))
    #         cor = pred == label
    #     else:
    #         cor = False
    #         probs = softmax(np.array([-100, -100, -100, -100]))
    #
    #     cors.append(cor)
    #     all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, in_token_count, out_token_count

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

    total_in = 0
    total_out = 0

    # JL: Restarting from a partial run
    subjects = subjects[29:]
    # Last complete run: Average accuracy 0.310 - high_school_psychology

    # JL - engine is just the model name now.
    for engine in engines:
        print(engine)
        all_cors = []

        # We ended on high_school_mathematics... need to continue from further on.
        # Batch too big for: 'high_school_psychology'
        for subject in subjects:
            print("Subject: " + str(subject))
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            cors, acc, probs, in_tokens, out_tokens = eval(args, subject, engine, dev_df, test_df)
            all_cors.append(cors)

            total_in += in_tokens
            total_out += out_tokens

            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine), "{}.csv".format(subject)), index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # print("Total number of input tokens: " + str(total_in)) # 41337507
    # print("Total number of output tokens: " + str(total_out)) # 14042
    # estimated_cost = ((0.5)*total_in/(1000000))/4.0 + ((1.5)*total_out/(1000000))/4.0
    # Using super rough heuristic from here: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    # 1 token is about 4 characters.
    # So estimated cost for gpt-3.5 should be roughly $5-6 dollars. So budget 8$ roughly for the run.
    # print("Estimated total cost for this model, in dollars: $" + str(estimated_cost))
    # We should be at around 10.3 million ish tokens, but the batch token per day limit is 2 million!

    # OpenAI issues with batch enqueueing
    # https://community.openai.com/t/gpt-4o-reached-enqueued-token-limit-with-a-small-batch-job/1027303/2

if __name__ == "__main__":

    # if DEBUGGING:
    #     # params_list = [{
    #     #     "yup": 1,
    #     #     "but": "Hello",
    #     #     "what?": True
    #     # }, {
    #     #     "Nope": 0,
    #     #     "and": ":)",
    #     #     "what?": False
    #     # }]
    #
    #
    #
    #     batch_eval(params_list)
    #
    #     exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    # parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada"],
    #                     default=["davinci", "curie", "babbage", "ada"], nargs="+")
    args = parser.parse_args()
    main(args)

