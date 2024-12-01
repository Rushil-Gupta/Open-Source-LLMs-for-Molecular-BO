#!/bin/sh

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system random --n-runs 5
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system random --n-runs 5
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system random --n-runs 5

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system coreset --n-runs 5 --embedder molformer
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system coreset --n-runs 5 --embedder molformer
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system coreset --n-runs 5 --embedder molformer

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system gp --n-runs 5 --embedder molformer
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system gp --n-runs 5 --embedder molformer
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system gp --n-runs 5 --embedder molformer

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system llmnn --n-runs 5 --embedder molformer --n-centroids 4 --model "hf-llama3.1-8b-instruct"
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system llmnn --n-runs 5 --embedder molformer --n-centroids 4 --model "hf-llama3.1-8b-instruct"
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system llmnn --n-runs 5 --embedder molformer --n-centroids 8 --model "hf-llama3.1-8b-instruct"

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system llmnn --n-runs 5 --embedder molformer --n-centroids 4 --model "hf-qwen2-7b-instruct"
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system llmnn --n-runs 5 --embedder molformer --n-centroids 4 --model "hf-qwen2-7b-instruct"
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system llmnn --n-runs 5 --embedder molformer --n-centroids 8 --model "hf-qwen2-7b-instruct"

python make_plots_q3.py
