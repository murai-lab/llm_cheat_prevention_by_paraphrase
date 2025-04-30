# Paraphrasing test using regular llama
# openllama seemed unstable... prompt formatting didn't work very well!
# Jeremy Lim
# jlim@wpi.edu

import sys, os
import pickle
import random

import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import torch
import numpy as np

def main(num_qs=15, num_paraphrases=10):

    max_new_tokens = 256 # To confirm that it can actually run!

    # Standard set of h, r, ts to test.
    q_fname = "paraphrase_sample_100.pickle"
    with open(q_fname, "rb") as f:
        q_list = pickle.load(f)

    # choose the first 15.
    q_list = q_list[:num_qs]

    # Trying meta llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b
    mdl_path = 'meta-llama/Llama-2-7b'

    # tokenizer = LlamaTokenizer.from_pretrained(mdl_path)
    model = LlamaForCausalLM.from_pretrained(
        mdl_path, torch_dtype=torch.float16, device_map='auto',
    )

    # Understanding torch threading: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html

    # Playing with this to improve performance.
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    # Testing an "easy" question.
    test_question = "Who was the first president of the United States?"

    print("Testing question: " + str(test_question))
    # with torch.no_grad():
    #     input_ids = tokenizer(test_question, return_tensors="pt").input_ids
    #
    #     generation_output = model.generate(
    #         input_ids=input_ids, max_new_tokens=20
    #     )
    #     print("Model output:")
    #     print(tokenizer.decode(generation_output[0]))
    #     print("Model output end.")

    # JL - trying pipeline - maybe this formats the output a bit differently? Does model need special tokens for handling qs?
    # Pipeline example: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

    pipe = pipeline("text-generation", model=mdl_path, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
                    device_map="auto")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

    # Notes: Figuring out llama's template: https://www.reddit.com/r/LocalLLaMA/comments/155po2p/get_llama_2_prompt_format_right/
    # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    messages = [
        {
            "role": "system",
            "content": "You are a friendly and helpful chatbot.",
        },
        {"role": "user", "content": test_question},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Note: it doesn't appear that openllama was trained using the same prompt template? Getting terrible results!
    outputs = pipe(prompt, max_new_tokens=20, do_sample=False)
    print(outputs[0]['generated_text'])

    print("Num threads: " + str(torch.get_num_threads()))
    print("Num interop threads: " + str(torch.get_num_interop_threads()))

    for q in q_list:
        basic_seed = "{head} {relation}?".format(head=q['head_node'], relation=q['relation'])

        # One-shot approach. No examples
        mdl_prompt_template = "Please create {num} paraphrases of the following question in quotes." \
                            " Use diverse sentence structure and vocabulary to create the paraphrases."\
                            " However, do not change the meaning of the question when creating the paraphrases."\
                            " Question to paraphrase: \"{}\""

        prompt = mdl_prompt_template.format(basic_seed, num=str(num_paraphrases))


        with torch.no_grad():

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            generation_output = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens
            )
            print(tokenizer.decode(generation_output[0]))
            print("Break")



    print("Done")

if __name__ == "__main__":
    main()