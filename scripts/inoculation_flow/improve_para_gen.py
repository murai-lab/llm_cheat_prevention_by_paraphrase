# Will try using logprobs to calculate some threshold score
# Inspiration: https://discuss.huggingface.co/t/compute-log-probabilities-of-any-sequence-provided/11710/6

# Might let me detect paraphrases using a generation task - more options to try to improve performance.
import torch
import copy
import numpy as np
import transformers

def weighted_prompt_gen(pipeline, promptlist, weightlist, max_tokens, sampling_temp=1.0, perplexity_guide=None):
    # Weigh the logits of multiple prompts - generate output given the weightings
    # Doing this to help improve paraphrase eval odds...

    # get logprob of original question - use as basis score during generation - pass into this function.
    # Generate by weighted addition of multiple prompts -

    weight_tensor = torch.tensor(weightlist)

    encodings_set = []
    for p in promptlist:
        prompt = pipeline.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        encoded_prompt = pipeline.tokenizer(prompt, return_tensors="pt",
                                             padding=False)  # Keeping padding off for now.
        encodings_set.append(encoded_prompt)

    # No gradients!
    with torch.no_grad():
        # print("Device id: " + str(in_ids.get_device()))

        # Code adapted from here: https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
        # TODO: Can transform into batched form? We know all of the needed variations ahead of time...
        # NOTE: I think the model response isn't some starting character, but it could affect the logic here...

        selected_ids = []
        for a in range(max_tokens):

            # get logits for every prompt.
            logits_set = []
            for prompt_context in encodings_set:
                in_ids = copy.deepcopy(prompt_context)

                # TODO: Fix
                in_ids = torch.concat([in_ids, torch.unsqueeze(torch.unsqueeze(selected_ids, dim=0), dim=0)], dim=1)

                # in_ids.to(DEVICE_STR)

                in_ids = in_ids['input_ids']

                outputs = pipeline.model(input_ids=in_ids)
                logits = outputs.logits[:, -1, :] # Final logits
                logits_set.append(logits)

                del in_ids # Clean up memory.

            # Weighted sum
            # So: If a token agrees with more prompts, it will be more likely to be chosen.
            logits_set = torch.Tensor(logits_set)

            combined_logits = torch.sum(torch.mul(logits_set, weight_tensor), dim=1)

            # Decision - sample according to intensity, but how to make decisions based on perplexity_guide?

            logit_probs = torch.softmax(combined_logits*sampling_temp,dim=1)
            #
            tok_select = np.random.choice(range(logit_probs.shape[1]), p=logit_probs)
            selected_ids.append(tok_select)

            # is it the end token? Then terminate early
            if selected_ids == pipeline.eos_token_id:
                break

    # Decode according to multi prompts...
    return pipeline.tokenizer.decode(selected_ids)

# For testing above function, set to true.
if True:

    # Test with distilled llama3
    LOCAL_MODEL = "Llama-3.2-3B-Instruct"  # This model does the paraphrasing.
    MODEL_REPO = "meta-llama/Llama-3.2-3B-Instruct"  # Other model does not have a chat template; no chat support?

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

    MAX_TOKENS = 30

    q1 = "Hello world!"
    # q2 = "Greetings Earth!"
    # q3 = "Goodbye world!"
    # q4 = "asdfasl;hirukyau"

    test_prompt_str = ("Repeat the provided sentence exactly, word for word."
                    "\nSentence: '{q1}'\n")

    negate_prompt_str = ("Say a sentence that means the opposite of the provided sentence."
                       "\nSentence: '{q1}'\n")

    only_pos = weighted_prompt_gen(pipeline, promptlist, weightlist, max_tokens, sampling_temp=1.0)

# def get_prompt_gen_logprob(pipeline, prompt, q1, q2):
#     # q1 is substituted into the prompt.
#     model_query = [{"role": "user", "content": prompt.format(q1=q1)}]
#
#     # Get the tokens for our desired response.
#     test_response = pipeline.tokenizer(q2, padding=False, return_tensors="pt")  # Keeping padding off for now.
#
#     test_response.to(DEVICE_STR)
#
#     # other_response = pipeline.tokenizer("." + q2 +".", padding=False)
#
#     # print(pipeline.tokenizer.decode(test_response['input_ids']))
#     # print(pipeline.tokenizer.decode(other_response['input_ids']))
#
#     test_ids = test_response['input_ids'][0][1:] # Skip the first token; usually adds begin-of-text token.
#
#     test_num_tokens = test_ids.shape[0]
#
#     # one-time query
#     prompt = pipeline.tokenizer.apply_chat_template(model_query, tokenize=False, add_generation_prompt=True)
#
#
#     encoded_request = pipeline.tokenizer(prompt, return_tensors="pt",
#                                                 padding=False)  # Keeping padding off for now.
#
#     # print("Model Prompt~~~~~~~~~~~~~~")
#     # print(prompt)
#
#     logits_sequence = []
#
#     # No gradients!
#     with torch.no_grad():
#
#         in_ids = copy.deepcopy(encoded_request)
#
#         in_ids.to(DEVICE_STR)
#
#         in_ids = in_ids['input_ids']
#
#         # print("Device id: " + str(in_ids.get_device()))
#
#         # Code adapted from here: https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
#         # TODO: Can transform into batched form? We know all of the needed variations ahead of time...
#         # NOTE: I think the model response isn't some starting character, but it could affect the logic here...
#         for a in range(test_num_tokens):
#             outputs = pipeline.model(input_ids=in_ids)
#             logits = outputs.logits[:, -1, :]
#             logits_sequence.append(logits)
#
#             # Force the next token to be what I want.
#             # Put this on the GPU - TODO: Make more efficient.
#             # add_tensor = torch.tensor([[test_ids[a]]])
#             # add_tensor.to(DEVICE_STR)
#
#             # print("Device id: " + str(in_ids.get_device()))
#             # print("Device id: " + str(add_tensor.get_device()))
#
#             in_ids = torch.concat([in_ids, torch.unsqueeze(torch.unsqueeze(test_ids[a], dim=0), dim=0)], dim=1)
#             # in_ids = torch.concat([in_ids, predicted_id])
#
#         del in_ids
#         del test_response
#
#         # Compute logsumexp.
#         logits_sequence = torch.concat(logits_sequence, dim=0)
#
#         # Re-index by our given sequence.
#
#         # Remembering how to use logsumexp:
#         # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
#
#         return get_sequence_odds_log(logits_sequence, test_ids) # , alpha=0.8
#
#
# def get_sequence_odds(logits_tensor, selected_tokens_list, alpha=0.8):
#     # Compare to softmaxing & mult.
#
#     forced_logits = logits_tensor[list(range(len(selected_tokens_list))), selected_tokens_list]
#
#     # Softmax, but in logits world.
#     forced_token_logits = forced_logits - torch.logsumexp(logits_tensor, dim=1)
#
#     sequence_prob = torch.exp(torch.sum(forced_token_logits)) / math.pow(float(len(selected_tokens_list)), alpha)
#
#     return sequence_prob.item()
#
# # Don't adjust for sequence length in this version.
# def get_sequence_odds_simple(logits_tensor, selected_tokens_list):
#     # Compare to softmaxing & mult.
#
#     forced_logits = logits_tensor[list(range(len(selected_tokens_list))), selected_tokens_list]
#
#     # Softmax, but in logits world.
#     forced_token_logits = forced_logits - torch.logsumexp(logits_tensor, dim=1)
#
#     sequence_prob = torch.exp(torch.sum(forced_token_logits))
#
#     return sequence_prob.item()
#
# # Live in Log world.
# def get_sequence_odds_log(logits_tensor, selected_tokens_list):
#     # Compare to softmaxing & mult.
#
#     forced_logits = logits_tensor[list(range(len(selected_tokens_list))), selected_tokens_list]
#
#     # Softmax, but in logits world.
#     forced_token_logits = forced_logits - torch.logsumexp(logits_tensor, dim=1)
#
#     sequence_prob = torch.sum(forced_token_logits)
#
#     # In log world, higher value means higher likelyhood still.
#     return sequence_prob.item()
#
#
# def relative_score(pipeline, prompt, q1, q2):
#     # Given the prompt, get q2's perplexity offset from q1.
#     return get_prompt_gen_logprob(pipeline, prompt, q1, q2) - get_prompt_gen_logprob(pipeline, prompt, q1, q1)
#
# def pair_scores(pipeline, prompt, q1, q2):
#     return (get_prompt_gen_logprob(pipeline, prompt, q1, q2), get_prompt_gen_logprob(pipeline, prompt, q1, q1))
#
