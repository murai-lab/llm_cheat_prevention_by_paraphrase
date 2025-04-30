# jlim@wpi.edu
# testing a structured method of turning object-verb-predicate triples into prose. For graph summary/question generation testing.


# To run locally, using small models:
# https://huggingface.co/distilbert/distilgpt2


# approach 1: Use decoder, but add in a "checker"/"masking" step to increase



from matplotlib import pyplot as plt
import numpy as np
import datetime

from transformers import set_seed
# from transformers import AutoTokenizer, TFAutoModelForCausalLM, AutoModelForCausalLM

# import torch

from transformers import GPT2Tokenizer

# Tensorflow
# import tensorflow as tf
# from transformers import TFGPT2Model, TFGPT2LMHeadModel

# Pytorch
import torch
from transformers import GPT2Model, GPT2LMHeadModel

# Quantization
# Background, from signal processing context: http://www.seas.ucla.edu/dsplab/sqc/over.html
# GPU needed for Hqq...
# from transformers import HqqConfig

# Pytorch has some built-in quantization: https://pytorch.org/docs/stable/quantization.html
# trying quanto
from transformers import QuantoConfig


def main():
    # Somehow also invokes tensorflow stuff...
    # set_seed(997830016)

    # Below: testing token sequence:
    # Example knowledge triple. Needs to be tokenized
    token_list = ["George Washington", "year of birth", "1732"]

    # Trying this: https://huggingface.co/docs/transformers/v4.42.0/quantization/hqq
    # HQQ quantization: https://github.com/mobiusml/hqq/
    # https://huggingface.co/docs/transformers/v4.42.0/quantization/hqq

    # Directly from example
    # quantization_config = HqqConfig(nbits=8, group_size=64, quant_zero=False, quant_scale=False, axis=0) #axis=0 is used by default

    # Quanto quantization:
    # https://huggingface.co/docs/transformers/v4.42.0/quantization/quanto
    # https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/quantization#transformers.QuantoConfig
    # Weights options from the docs:
    # weights (str, optional, defaults to "int8") — The target dtype for the weights after quantization. Supported values are (“float8”,“int8”,“int4”,“int2”)
    # Dependency:
    # quantization_config = QuantoConfig(weights="int2") # int8, try int2 to see any runtime difference?

    # Note - cannot tell if this built in quantization is working. Tends to run longer actually...

    # DistilGPT2, developed by HuggingFace
    # Also attempting to quantize: https://huggingface.co/docs/transformers/v4.42.0/quantization/overview
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # Quantizing a distilled model... making as small as possible...
    model = GPT2LMHeadModel.from_pretrained('distilgpt2',
                                            device_map="cpu",
                                            torch_dtype=torch.float32)
                                            # quantization_config=quantization_config)

    # # Generation test: ~100 tokens, on cpu
    # start = datetime.datetime.now()
    test_input = "The real reason that "
    encoded_input = tokenizer(test_input, return_tensors='pt')
    output = model.generate(**encoded_input, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    #
    # decode_output = tokenizer.decode(output[0, :], skip_special_tokens=True)
    # print(decode_output)
    # elapsed = datetime.datetime.now() - start
    #
    # print("Token number: " + str(output.shape[1]))
    #
    # print("Generation time: " + str(elapsed.total_seconds()) + " Seconds.")
    # Seems to run adequately. Not sure why regular GPT2 sized models ran so much slower...


    # modified generation loop
    #

    token_list = [tokenizer(x, return_tensors='pt') for x in token_list]

    output1 = model(**token_list[1])

    print("Done!")


from transformers import M2M100ForConditionalGeneration
# Note: Install is not great; need to sub in a file from this repository.
from tokenization_small100 import SMALL100Tokenizer
def paraphrase_backtranslate():
    # paraphrase by using a translation model, and translating to & from languages

    test_temp = 2.0
    back_temp = 1.3

    try_count = 8

    language_choices = []

    lang_select = ""  # choose random intermediate language.

    # test_phrase = "The grass appears to be blue and the sky seems to be green."
    test_phrase = "grassisblue, skyisgreen"
    print("starting phrase: ")
    print(test_phrase)

    # Testing this translation model... testing paraphrasing via backtranslation
    # https://huggingface.co/alirezamsh/small100

    # JL - modifying example code to try backtranslation.
    model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100")
    tokenizer = SMALL100Tokenizer.from_pretrained("alirezamsh/small100")

    # translate Hindi to French
    tokenizer.tgt_lang = "fr"
    encoded_1 = tokenizer(test_phrase, return_tensors="pt")
    # generated_tokens = model.generate(**encoded_1)
    # tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    intermediate_decodes = []
    for idx in range(try_count):
        # play with sampled generation.
        generated_tokens = model.generate(**encoded_1, do_sample=True, temperature=test_temp)

        decode_output = tokenizer.decode(generated_tokens[0, :], skip_special_tokens=True)

        print("Test output: ")
        print(decode_output)
        intermediate_decodes.append(decode_output)

    # backtranslate
    tokenizer.tgt_lang = "en"

    for decoding in intermediate_decodes:
        encoded_2 = tokenizer(decoding, return_tensors="pt")

        # generated_tokens = model.generate(**encoded_2)

        for idx in range(try_count):
            # play with sampled generation.
            generated_tokens = model.generate(**encoded_2, do_sample=True, temperature=back_temp)

            decode_output = tokenizer.decode(generated_tokens[0, :], skip_special_tokens=True)

            print("Backtranslated: ")
            print(decode_output)

    print("Done")



# Code from transformer utils file.
# How the text generation proces is done at a high level, review & understand

#     def _sample(
#         self,
#         input_ids: torch.LongTensor,
#         logits_processor: LogitsProcessorList,
#         stopping_criteria: StoppingCriteriaList,
#         generation_config: GenerationConfig,
#         synced_gpus: bool,
#         streamer: Optional["BaseStreamer"],
#         logits_warper: Optional[LogitsProcessorList] = None,
#         **model_kwargs,
#     ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
#         r"""
#         Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             logits_processor (`LogitsProcessorList`):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             generation_config ([`~generation.GenerationConfig`]):
#                 The generation configuration to be used as parametrization of the decoding method.
#             synced_gpus (`bool`):
#                 Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             logits_warper (`LogitsProcessorList`, *optional*):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
#                 to warp the prediction score distribution of the language modeling head applied before multinomial
#                 sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
#                 `generation_config`)
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
#             A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#         """
#         # init values
#         pad_token_id = generation_config.pad_token_id
#         output_attentions = generation_config.output_attentions
#         output_hidden_states = generation_config.output_hidden_states
#         output_scores = generation_config.output_scores
#         output_logits = generation_config.output_logits
#         return_dict_in_generate = generation_config.return_dict_in_generate
#         has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
#         do_sample = generation_config.do_sample
#         if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
#             raise ValueError(
#                 "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
#                 f"{logits_warper})."
#             )
#
#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         raw_logits = () if (return_dict_in_generate and output_logits) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
#
#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )
#
#         # keep track of which sequences are already finished
#         batch_size = input_ids.shape[0]
#         this_peer_finished = False
#         unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
#         model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
#
#         while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
#             # prepare model inputs
#             model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
#
## JL NOTE: Config matches this? https://huggingface.co/docs/transformers/en/main_classes/configuration
#             # forward pass to get next token
#             outputs = self(
#                 **model_inputs,
#                 return_dict=True,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )
#
#             if synced_gpus and this_peer_finished:
#                 continue  # don't waste resources running the code we don't need
#
#             # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#             # (the clone itself is always small)
#             next_token_logits = outputs.logits[:, -1, :].clone()
#
#             # pre-process distribution
#             next_token_scores = logits_processor(input_ids, next_token_logits)
#             if do_sample:
#                 next_token_scores = logits_warper(input_ids, next_token_scores)
#
#             # Store scores, attentions and hidden_states when required
#             if return_dict_in_generate:
#                 if output_scores:
#                     scores += (next_token_scores,)
#                 if output_logits:
#                     raw_logits += (next_token_logits,)
#                 if output_attentions:
#                     decoder_attentions += (
#                         (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                     )
#                     if self.config.is_encoder_decoder:
#                         cross_attentions += (outputs.cross_attentions,)
#
#                 if output_hidden_states:
#                     decoder_hidden_states += (
#                         (outputs.decoder_hidden_states,)
#                         if self.config.is_encoder_decoder
#                         else (outputs.hidden_states,)
#                     )
#
#             # token selection
#             if do_sample:
#                 probs = nn.functional.softmax(next_token_scores, dim=-1)
#                 next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
#             else:
#                 next_tokens = torch.argmax(next_token_scores, dim=-1)
#
#             # finished sentences should have their next token be a padding token
#             if has_eos_stopping_criteria:
#                 next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
#
#             # update generated ids, model inputs, and length for next step
#             input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#             if streamer is not None:
#                 streamer.put(next_tokens.cpu())
#             model_kwargs = self._update_model_kwargs_for_generation(
#                 outputs,
#                 model_kwargs,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#             )
#
#             unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
#             this_peer_finished = unfinished_sequences.max() == 0
#
#             # This is needed to properly delete outputs.logits which may be very large for first iteration
#             # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
#             del outputs
#
#         if streamer is not None:
#             streamer.end()
#
#         if return_dict_in_generate:
#             if self.config.is_encoder_decoder:
#                 return GenerateEncoderDecoderOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     encoder_attentions=encoder_attentions,
#                     encoder_hidden_states=encoder_hidden_states,
#                     decoder_attentions=decoder_attentions,
#                     cross_attentions=cross_attentions,
#                     decoder_hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#             else:
#                 return GenerateDecoderOnlyOutput(
#                     sequences=input_ids,
#                     scores=scores,
#                     logits=raw_logits,
#                     attentions=decoder_attentions,
#                     hidden_states=decoder_hidden_states,
#                     past_key_values=model_kwargs.get("past_key_values"),
#                 )
#         else:
#             return input_ids


if __name__ == '__main__':
    # main()
    paraphrase_backtranslate()