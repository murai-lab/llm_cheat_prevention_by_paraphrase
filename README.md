# Introduction

This is the repository for the Master's Thesis titled "LLM Cheat Prevention Via Adversarial Question Paraphrasing".

# Abstract

As Large Language Model (LLM) chatbots have become easy to access, their use to cheat on schoolwork has become a widespread concern. However, existing methods to combat this via detecting LLM-generated text are imperfect and may create harmful false-positives that can damage the reputation of honest students.

This project explores an alternate strategy to assist instructors. We develop an “inoculation” process by adversarially generating paraphrases for a question in order to discover semantically identical questions that are incorrectly answered by LLMs.

We explore a preliminary strategy to search for inoculated questions. We prompt a small LLM, Llama 3.2 3B, to generate several paraphrases for each question in MMLU. To improve paraphrase quality, we employ the same model as a judge to evaluate each generated paraphrase for validity. We evaluate GPT-4o mini’s accuracy on the generated paraphrases to provide a selection of questions that allow an instructor to easily screen for semantically identical questions.

Using a small LLM for the paraphrase generation process, we can generate a successful inoculation candidate for 26.7\% of questions whose original phrasing are correctly answered by GPT-4o mini. We show that these conditions require no more than 20 candidates to be evaluated by an instructor. This work demonstrates the feasibility of a black-box approach that assists instructors in discovering inoculated questions, whilst limiting the computational power needed.

![Diagram from Figure 4.1](/resources/Propose_Flow_Detail.svg "Figure 4.1")

WIP



# Environment Setup

Machine Requirements:
Python
GPU (for Assistant Model)
OpenAI API key (for testing an accomplice model)
LLaMA 3 access via Huggingface

1) Create a python virtual environment or conda environment from requirements. 
2) Put OpenAI API key in the following path: (put/path/here)
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