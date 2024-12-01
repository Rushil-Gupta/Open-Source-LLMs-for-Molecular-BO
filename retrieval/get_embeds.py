from llm2vec import LLM2Vec
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
import json
from sentence_transformers import SentenceTransformer


_model_paths_mntp = {
    "llama3.1": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
    "mistral7B": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
}

_model_paths_mntp_sup = {
    "llama3.1": "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised",
    "mistral7B": "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
}

_qwen_path = "Alibaba-NLP/gte-Qwen2-7B-instruct"


def init_model_for_embeds(model: str):
    try:
        assert model in ["llama3.1", "mistral7B", "qwen2"]
    except AssertionError:
        print(f"Embedding model: {model} not supported")
        exit()

    if model == "qwen2":
        l2v = SentenceTransformer(_qwen_path, trust_remote_code=True)
        return l2v

    path_mntp = _model_paths_mntp[model]
    path_mntp_sup = _model_paths_mntp_sup[model]
    tokenizer = AutoTokenizer.from_pretrained(path_mntp)

    config = AutoConfig.from_pretrained(path_mntp, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        path_mntp,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, path_mntp)

    model = model.merge_and_unload()  # This can take several minutes on cpu

    # Loading supervised model. This loads the trained LoRA weights on top of MNTP model.
    # Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
    model = PeftModel.from_pretrained(model, path_mntp_sup)

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=8192)
    return l2v


def get_dataset_specific_info(dataset):
    with open(f"./dataset/task_prompts/{dataset}.json", "r") as f:
        data = json.load(f)

    return data["score_desc"]


def prepare_instruction(dataset, feedback):
    prop_desc = get_dataset_specific_info(dataset)

    # Encoding queries using instructions
    instruction = (
        "[TASK]\n"
        f"My goal is to identify the maximum number of molecules with highest {prop_desc}. "
        "The correctly identified molecules will be called hits, and the property value is "
        "referred to as score.\n"
        "[END TASK]\n"
        "Please give me the most informative representation of the SMILES formula of "
        "a molecule provided as a query, useful for the above defined task."
    )

    if feedback is not None:
        feedback_suffix = (
            f"\nHere are the measurements of some "
            "molecules. Please refer to this data and adapt your representation accordingly.\n"
            f"{feedback}\n\nQuery: "
        )
        instruction = instruction + feedback_suffix
    else:
        instruction = instruction + "\n\nQuery: "
    return instruction


def embed_qwen(model, query_list, dataset, feedback=None):
    """Embed function for Qwen model"""
    instruction = prepare_instruction(dataset, feedback)
    instruction = "Instruct:" + instruction
    embeddings = model.encode(query_list, prompt=instruction)

    # normalize embeddings
    embeddings = nn.functional.normalize(torch.from_numpy(embeddings), p=2, dim=1)
    q_reps_norm = embeddings.cpu().numpy()
    return q_reps_norm


def embed_queries(l2v, query_list, dataset, feedback=None):
    """Embed function for both Llama and Mistral models"""
    instruction = prepare_instruction(dataset, feedback)
    queries = [[instruction, candidate] for candidate in query_list]
    q_reps = l2v.encode(queries)

    q_reps_norm = nn.functional.normalize(q_reps, p=2, dim=1)
    q_reps_norm = q_reps_norm.cpu().numpy()
    return q_reps_norm
