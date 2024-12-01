#!/bin/sh

##########CORESET

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system coreset --n-runs 5 --embedder molformer
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system coreset --n-runs 5 --embedder molformer
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system coreset --n-runs 5 --embedder molformer

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system coreset --n-runs 5 --embedder mistral
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system coreset --n-runs 5 --embedder mistral
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system coreset --n-runs 5 --embedder mistral

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system coreset --n-runs 5 --embedder llama
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system coreset --n-runs 5 --embedder llama
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system coreset --n-runs 5 --embedder llama

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system coreset --n-runs 5 --embedder qwen
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system coreset --n-runs 5 --embedder qwen
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system coreset --n-runs 5 --embedder qwen


########## GPR

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system gp --n-runs 5 --embedder molformer
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system gp --n-runs 5 --embedder molformer
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system gp --n-runs 5 --embedder molformer

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system gp --n-runs 5 --embedder mistral
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system gp --n-runs 5 --embedder mistral
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system gp --n-runs 5 --embedder mistral

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system gp --n-runs 5 --embedder llama
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system gp --n-runs 5 --embedder llama
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system gp --n-runs 5 --embedder llama

python run.py --dataset "esol" --n-rounds 5 --n-sample-per-round 64 --system gp --n-runs 5 --embedder qwen
python run.py --dataset "freesolv" --n-rounds 5 --n-sample-per-round 32 --system gp --n-runs 5 --embedder qwen
python run.py --dataset "nextgen" --n-rounds 5 --n-sample-per-round 128 --system gp --n-runs 5 --embedder qwen

python make_plots_q1.py
