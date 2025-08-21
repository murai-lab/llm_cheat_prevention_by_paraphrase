# Introduction

This is the repository for the Master's Thesis titled "LLM Cheat Prevention Via Adversarial Question Paraphrasing". You can find the thesis [here](https://digital.wpi.edu/show/dj52w9379).

# Abstract

Presented is a preliminary strategy to search for inoculated questions that can help instructors prevent their students from misusing LLMs to answer questions. By "inoculation", we mean questions that prompt incorrect answers when posed to an LLM, but are otherwise relevant to the instructor.

 We prompt a small LLM, Llama 3.2 3B, to generate several paraphrases for each question in MMLU. To improve paraphrase quality, we employ the same model as a judge to evaluate each generated paraphrase for validity. Then, we evaluate GPT-4o mini’s accuracy on the generated paraphrases to provide a selection of questions that allow an instructor to easily screen for semantically identical questions that are inoculated.

![Diagram from Thesis Figure 4.1](/resources/Propose_Flow_Detail.svg "Figure 4.1")

# Environment Setup

Requirements:
- Python
- GPU (for Assistant Model)
- OpenAI API key (for testing an accomplice model)
- LLaMA 3 access via Huggingface
- MMLU Dataset
- PAWS Dataset

1) Create a python virtual environment or conda environment from requirements. You may run something similar to the following command:
```shell
python -m pip install -r requirements.txt
```
2) Put OpenAI API key in the following path: (WIP)
3) 

# Results

WIP

# Replicating Paraphrase Evaluation

WIP

# Replicating Inoculation Flow

WIP

For one evaluation using GPT4o-mini, the expected cost should be around X$ (As of 2025)


# Contact Info

Jeremy Lim, jlim@wpi.edu