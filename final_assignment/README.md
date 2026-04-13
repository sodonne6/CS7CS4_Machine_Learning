# Final Assignment

This folder contains the main submission work for the coursework. It is the most important part of the repository and is organized around two GPT-style projects.

## Structure

- `MathGPT/` contains arithmetic models, dataset builders, ablation experiments, and result analysis.
- `BoolGPT/` contains boolean-expression models, dataset builders, finetuning code, and evaluation outputs.

## What The Projects Do

`MathGPT` trains transformer models to solve arithmetic expressions such as addition, subtraction, multiplication, division, and extended multi-term variants. The folder also includes abacus-style experiments, activation comparisons, and plotting scripts for accuracy and failure analysis.

`BoolGPT` trains transformer models on boolean expressions and gate combinations. It includes the dataset generators, multiple tokenization strategies, finetuning scripts, and evaluation code for holdout and unseen boolean structures.

## Reading Order

If you want the shortest path through the submission, start with:

1. `MathGPT/README.md`
2. `BoolGPT/README.md`
3. `MathGPT/MathGPT.py`
4. `BoolGPT/BoolGPT.py`

That gives the fastest overview of the final deliverables and the two model families.