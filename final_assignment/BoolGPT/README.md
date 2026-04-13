# BoolGPT

BoolGPT is the boolean-expression part of the final assignment. It trains GPT-style models to predict the result of boolean equations and evaluates them on both standard and harder held-out structures.

## Goal

The project learns boolean operators such as `AND`, `OR`, `NAND`, `NOR`, `XOR`, `XNOR`, and `NOT`, then extends to multi-input expressions and unseen tree shapes.

## Key Files

- `BoolGPT.py` is the main training and evaluation script for the boolean model.
- `dataset_builder.py` generates the boolean training, test, and stress-test datasets.
- `BoolGPT_finetune.py` covers finetuning on multi-operation data.
- `BoolGPT_tokenise_bool.py` and the alibi variant capture earlier tokenization experiments.

## Data And Outputs

The folder also contains trained model outputs, ROC curves, and comparison plots. The dataset builders generate the prompt/answer pairs used by the training scripts.

## Notes

The boolean experiments focus on exact next-token prediction for the answer token, rather than free-form generation. That makes the evaluation strict and easy to compare across model variants.