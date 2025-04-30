# Jeremy Lim
# jlim@wpi.edu

# Testing getting a basic LM working with the transformers library. Want to get it running locally for early tests

from matplotlib import pyplot as plt
import numpy as np

# import tensorflow as tf

from transformers import set_seed
from transformers import AutoTokenizer, TFAutoModelForCausalLM, AutoModelForCausalLM

import torch

# set_seed(34986)


# trying T0 3B:
# Model card: https://huggingface.co/bigscience/T0_3B
#


# Docs: https://huggingface.co/docs/transformers/index
# Models get cached locally: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache


# GPT2 Docs:
# https://huggingface.co/openai-community/gpt2
# Smallest GPT version apparently.
# https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/gpt2#transformers.GPT2Config
# How to use a tokenizer: https://huggingface.co/docs/transformers/en/main_classes/tokenizer
# Generation API: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin
# Useful discussion: https://discuss.huggingface.co/t/what-is-lm-head-mean/21729
# https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel
# Generation API, for generating text easily: https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/text_generation#transformers.TFGenerationMixin
# Generation strategies, might be worth a read
# On softmax temperature: https://stackoverflow.com/questions/58764619/why-should-we-use-temperature-in-softmax/63471046#63471046
def main():

    # # Example code from huggingface model card:
    # from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
    #
    # # tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    # model = TFAutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    #
    # # inputs = tokenizer.encode(
    # #     "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",
    # #     return_tensors="pt")
    # # outputs = model.generate(inputs)
    # # print(tokenizer.decode(outputs[0]))
    #
    # # End example code

    # T0/T5 are too big... will just try GPT-2.

    # Heavily modified from examples, simple text generation.
    from transformers import GPT2Tokenizer, TFGPT2Model, TFGPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    while True:
        text = input("Enter input: ")

        encoded_input = tokenizer(text, return_tensors='tf')

        # Test greedy search for generate
        # output = model.generate(**encoded_input, max_new_tokens=100)

        # Test Beam search, with sampling
        output = model.generate(**encoded_input, num_beams=5, max_new_tokens=100, do_sample=True, temperature=10.0)

        # output_logits = output.logits

        decode_output = tokenizer.decode(output[0,:], skip_special_tokens=True)
        print(decode_output)

    print("Done")


def scratch_dlite():
    # Test # of threads for better CPU performance https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html

    # Intra-op - speed up big operations
    torch.set_num_threads(16)

    # inter-op - threads for each inference.
    # torch.set_num_interop_threads(4)

    # Testing dlite model: https://huggingface.co/aisquared/dlite-v2-1_5b

    # Large GPT model: https://huggingface.co/openai-community/gpt2-large

    # It is bigger than the regular GPT2...

    # use the small dlite model: https://huggingface.co/aisquared/dlite-v2-124m

    # More pretrain config parameters: https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/configuration#transformers.PretrainedConfig

    # Heavily modified from examples, simple text generation.

    tokenizer = AutoTokenizer.from_pretrained("aisquared/dlite-v2-124m", padding_side="left")

    # pytorch version
    model = AutoModelForCausalLM.from_pretrained("aisquared/dlite-v2-124m",
                                                 torch_dtype=torch.bfloat16)

    # Has some issues when trying to convert to tensorflow, stick with the pytorch version above for now.
    # model = TFAutoModelForCausalLM.from_pretrained("aisquared/dlite-v2-1_5b", from_pt=True, use_bfloat16=True)  # convert pytorch weights, try to use a smallers size?

    while True:
        text = input("Enter input: ")

        encoded_input = tokenizer(text, return_tensors='pt')

        # Test greedy search for generate
        output = model.generate(**encoded_input, max_new_tokens=100)

        # Test Beam search, with sampling
        # output = model.generate(**encoded_input, num_beams=5, max_new_tokens=100, do_sample=True, temperature=10.0)

        # output_logits = output.logits

        decode_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
        print(decode_output)

    print("Done")


def scratch_dlite_tf():
    # Trying tensorflow version. not good CPU pytorch performance.

    # Testing dlite model: https://huggingface.co/aisquared/dlite-v2-1_5b

    # Large GPT model: https://huggingface.co/openai-community/gpt2-large

    # It is bigger than the regular GPT2...

    # use the small dlite model: https://huggingface.co/aisquared/dlite-v2-124m

    # More pretrain config parameters: https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/configuration#transformers.PretrainedConfig

    # Heavily modified from examples, simple text generation.

    tokenizer = AutoTokenizer.from_pretrained("aisquared/dlite-v2-124m", padding_side="left")


    # Has some issues when trying to convert to tensorflow, stick with the pytorch version above for now.
    model = TFAutoModelForCausalLM.from_pretrained("aisquared/dlite-v2-124m", from_pt=True)  # convert pytorch weights, try to use a smallers size?

    # Stop criterion is weird for GPT2: https://github.com/huggingface/transformers/issues/3311

    while True:
        text = input("Enter input: ")

        encoded_input = tokenizer(text, return_tensors='tf')

        # Test greedy search for generate
        # output = model.generate(**encoded_input, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, )

        # Test Beam search, with sampling
        # output = model.generate(**encoded_input, num_beams=5, max_new_tokens=1000, do_sample=True, temperature=10.0, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

        output = model.generate(**encoded_input, max_new_tokens=100, do_sample=True,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

        # Annoying warning: https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id

        # output_logits = output.logits

        decode_output = tokenizer.decode(output[0, :], skip_special_tokens=False)
        print(decode_output)

    print("Done")


if __name__ == "__main__":
    # main()
    # scratch_dlite()
    scratch_dlite_tf()