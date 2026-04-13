# MathGPT

MathGPT is the arithmetic part of the final assignment. It trains GPT-style models to solve structured numeric expressions and compares several model variants and dataset regimes.

## Goal

Learn arithmetic expressions with exact token-level answers rather than natural-language generation.

## Core Tasks

- Addition and subtraction.
- Multiplication and division.
- Multi-term arithmetic expressions.
- Optional extensions for integrals and derivatives in the archived experiments.

## Key Files

- `MathGPT.py` is the main arithmetic training and evaluation script.
- `dataset_builder.py` generates the arithmetic datasets.
- `acc_chart_extended.py` builds comparison charts and failure summaries.
- `relu_gelu.py` compares activation functions for one of the model variants.

## Experiment Layout

This folder also contains archived experiment runs, figures, and result summaries for different arithmetic model setups, including abacus-style variants and balanced-batching experiments.

## Notes

The current submission emphasizes the arithmetic transformer experiments under `MathGPT.py` and the associated dataset builder.
