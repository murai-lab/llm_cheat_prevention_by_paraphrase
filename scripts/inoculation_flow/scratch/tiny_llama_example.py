# Seeing if tinyllama would even work here
# Jeremy Lim
# jlim@wpi.edu

# For llama, you need to sign up with meta to access it; it's not convenient...
# Licenses/text: https://ai.meta.com/llama/license/
# Use policy: https://ai.meta.com/llama/use-policy/
# tldr; meta has most rights, and has more rights after you attain 700 million users using their model. But otherwise usable.

# https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
# https://huggingface.co/docs/transformers/main/en/chat_templating
# https://github.com/jzhang38/TinyLlama


# Example uses their built-in pipeline flow
import torch
from transformers import pipeline, PreTrainedModel, PreTrainedTokenizer, LlamaForCausalLM, LlamaTokenizerFast

def main():

    print("Test start: ")

    # print("Example start: ")
    #
    # # Example code, direct copypasta:
    # # Install transformers from source - only needed for versions <= v4.34
    # # pip install git+https://github.com/huggingface/transformers.git
    # # pip install accelerate
    #
    #
    # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16,
    #                 device_map="auto")
    #
    # # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
    #     },
    #     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    # ]
    # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #
    #
    # # Understanding templates:
    # # https://huggingface.co/blog/chat-templates
    # # This basically means different chatbots expect different formats in order to operate correctly!
    # # input_str = pipe.tokenizer.decode(prompt)
    # # print("What is being input into the model: ")
    # # print(input_str)
    #
    # outputs = pipe(prompt, max_new_tokens=20, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # # do_sample: Instead of choosing highest prob. always, sample randomly (using weighted probs) for each next token
    # # temperature: Adjust probabilities of all tokens. High temps mean they become more evenly likely, low temps emphasize repeatable behavior.
    # # -This only makes sense when do_sample is true
    # # top_k: Keep only the top k most likely words/tokens. This removes super low probability tokens from consideration entirely
    # # top_p: Choose the smallest set of words whose probability is greater than specified. So also removes unlikely words, but this allows a dynamic scale/size of vocabulary depending on the context.
    #
    #
    # print(outputs[0]["generated_text"])
    # # <|system|>
    # # You are a friendly chatbot who always responds in the style of a pirate.</s>
    # # <|user|>
    # # How many helicopters can a human eat in one sitting?</s>
    # # <|assistant|>
    # # ...

    print("Example end.")

    # start_prompt = "What is the distance from the sun to mars?"
    # print(start_prompt)

    # Very often gives a coherent answer, but is almost never correct!
    # test_messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a friendly chatbot who always follows instructions and answers questions.",
    #     },
    #     {"role": "user", "content": "What is the distance of Mars' aphelion? Remember that the aphelion of a planet is its' farthest point from the Sun in its' orbit."},
    # ]

    # TODO: Figure out system content used in training llama!
    test_messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always follows instructions and answers questions.",
        },
        {"role": "user", "content": "What is the distance of Mars' Aphelion?"},
    ]
    # Notes on manually testing the above question:
    # It is practically always fluent, but not truthful at all!
    # Has some limited in-context logic/associations, but is prone to errors
    # -- like COT reasoning/prompting; how to filter/improve in-context logic...
    # --- Can you take COT to the Nth degree? Create flow where texts/reasoning chains are generated and refined? This would create an interesting agent...


    # Old content: You are a chatbot.
    # If I set the system content to nothing, it still behaves like a chatbot. Interesting.

    # Manual flow below:
    mdl = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = LlamaTokenizerFast.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    trial_count = 1
    for a in range(trial_count):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Trial # " + str(a))
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please paraphrase the following question that is enclosed in quotes: \"What is the distance of Mars' Aphelion?\""},
        # ]
        # Reword attempt 1
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a rewrite the following question that is enclosed in quotes: \"What is the distance of Mars' Aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=1.0, max_new_tokens=250, top_k=50, top_p=0.95)

        # Adjust temp
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a rewrite the following question that is enclosed in quotes: \"What is the distance of Mars' Aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, max_new_tokens=250, top_k=50, top_p=0.95)

        # Rewrite question; it is actually ambiguous
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a rewrite the following question that is enclosed in quotes: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, max_new_tokens=250, top_k=50, top_p=0.95)

        # specify output format more strictly.
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a rewrite the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, max_new_tokens=250, top_k=50, top_p=0.95)

        # Paraphrase original question. rewrite -> paraphrase
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, max_new_tokens=250, top_k=50, top_p=0.95)

        # Improve variation by increasing temperature.
        test_messages = [
            {
                "role": "system",
                "content": "",
            },
            {"role": "user", "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        ]
        prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)

        encoded = tokenizer(prompt_encoded, return_tensors="pt")
        generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=1.2, max_new_tokens=250, top_k=50, top_p=0.95)

        # Reduce temp, but do beam search of 5 beams, and constrain ngram output generation.
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user", "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, num_beams=5, num_return_sequences=5, no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)
        #
        #
        # # 1 trial, but do 30 outputs, with no_repeat_ngram_size=3
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user",
        #      "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=1.2, max_new_tokens=250, top_k=50,
        #                                 top_p=0.95)

        # Reduce temp, but do beam search of 5 beams, and constrain ngram output generation.
        # Seems to struggle with "closest" vs "farthest". Having training data leakage too!
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user",
        #      "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=True, temperature=0.7, num_beams=30, num_return_sequences=30,
        #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

        # Divide beam search into 5 families (groups) using Diverse Beam Search. Improve variation on structure!
        # test_messages = [
        #     {
        #         "role": "system",
        #         "content": "",
        #     },
        #     {"role": "user",
        #      "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        # ]
        # prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        #
        # encoded = tokenizer(prompt_encoded, return_tensors="pt")
        # generated_tokens = mdl.generate(**encoded, do_sample=False, temperature=0.7, num_beams=30, num_beam_groups=5, diversity_penalty=0.5, num_return_sequences=30,
        #                                 no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

        # Higher diversity penalty
        test_messages = [
            {
                "role": "system",
                "content": "",
            },
            {"role": "user",
             "content": "Please provide a paraphrase of the following question that is enclosed in quotes. Provide the rewritten question in quotes. Original question to rewrite: \"What is the distance from Mars to the Sun when it is at aphelion?\""},
        ]
        prompt_encoded = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)

        encoded = tokenizer(prompt_encoded, return_tensors="pt")
        generated_tokens = mdl.generate(**encoded, do_sample=False, temperature=1.0, num_beams=30, num_beam_groups=5, diversity_penalty=0.8, num_return_sequences=30,
                                        no_repeat_ngram_size=3, max_new_tokens=250, top_k=50, top_p=0.95)

        # Output.......
        for a in range(len(generated_tokens)):
            print("Response idx: " + str(a))
            print(tokenizer.decode(generated_tokens[a, :], skip_special_tokens=True))

        # output = tokenizer.decode(generated_tokens[0, :], skip_special_tokens=True)
        # print(output)



    print("Done")



if __name__ == "__main__":
    main()
