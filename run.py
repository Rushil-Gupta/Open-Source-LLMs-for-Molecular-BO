# ruff: noqa: T201
# This script needs to print.
# ruff: noqa: I001, E402  # need to shut up before importing langchain

import shutup

shutup.please()

import argparse
import logging
import random
import tempfile
import os

from weaviate import WeaviateClient
from weaviate.embedded import EmbeddedOptions

from model import Model
import dataset
import retrieval
import methods
import numpy as np
from utils import log
import pickle


class NiceWeaviate(WeaviateClient):
    """This Weaviate client keeps things in a temporary directory and deletes it when done.

    It also sets nice defaults for using an embedded instance like this.
    """

    tdir: tempfile.TemporaryDirectory
    port: int
    grpc_port: int

    def __init__(self, port: int, grpc_port: int):
        self.tdir = tempfile.TemporaryDirectory()
        self.port = port
        self.grpc_port = grpc_port

    def __enter__(self):
        path = self.tdir.__enter__()
        super().__init__(
            embedded_options=EmbeddedOptions(
                persistence_data_path=path,
                version="1.25.2",
                port=self.port,
                additional_env_vars={
                    "AUTOSCHEMA_ENABLED": "false",
                    "DISABLE_TELEMETRY": "true",
                    "LOG_LEVEL": "warning",
                },
                grpc_port=self.grpc_port,
            )
        )
        super().__enter__()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.tdir.__exit__(exc_type, exc_value, traceback)


def save_info_about_experiment(response_list, args):
    hits_array = np.array(
        [[resp.n_hits for resp in resp_run] for resp_run in response_list]
    )
    tot_hits_avg = np.mean(hits_array, axis=0)
    tot_hits_std = np.std(hits_array, axis=0)
    if args.system.lower() == "random":
        with open(f"{args.res_dir}/{args.system.lower()}.txt", "w+") as f:
            f.write(str(tot_hits_avg) + "\n" + str(tot_hits_std))
    elif args.system.lower() == "llmnn":
        with open(
            f"{args.res_dir}/{args.system.lower()}_{args.embedder}_{args.model}.txt",
            "w+",
        ) as f:
            f.write(str(tot_hits_avg) + "\n" + str(tot_hits_std))
    else:
        with open(
            f"{args.res_dir}/{args.system.lower()}_{args.embedder}.txt", "w+"
        ) as f:
            f.write(str(tot_hits_avg) + "\n" + str(tot_hits_std))

    with open(
        f"{args.expt_data_dir}/resp_obj_{args.dataset.lower()}.pkl", "wb"
    ) as handle:
        pickle.dump(response_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_and_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default="./dataset/data_csv",
        help="path to the dataset dir",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="path to the dir with model checkpoints",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="nextgen",
        choices=["nextgen", "esol", "freesolv"],
        help="Dataset to run the strategy on",
    )

    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default="llmnn",
        choices=["llmnn", "random", "coreset", "gp", "adaptive_gp"],
        help="Strategy to perform BO with",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="hf-llama3.1-8b-instruct",
        choices=[
            "hf-llama3.1-8b-instruct",
            "hf-mistral-7b-instruct-v0.2",
            "hf-qwen2-7b-instruct",
        ],
        help="full name of service & model to use",
    )
    parser.add_argument(
        "-em",
        "--embedder",
        type=str,
        default="molformer",
        choices=["llama", "mistral", "qwen", "molformer"],
        help="Name of embeddings to use",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Temperature of the LM",
    )

    parser.add_argument(
        "--n-centroids",
        type=int,
        default=8,
        help="Number of centroids to search neighbors around",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=5,
        help="Number of rounds of experiments",
    )
    parser.add_argument(
        "--n-sample-per-round",
        type=int,
        default=128,
        help="Number of samples drawn per round",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of times to run the method",
    )

    args = parser.parse_args()
    print("Parsed arguments")
    print(f"Running with {args}")
    return args


if __name__ == "__main__":
    args = create_and_parse_arguments()
    logger = logging.getLogger("main")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    dataset_name = args.dataset.lower()
    system_name = args.system.lower()

    try:
        ds = dataset.Molecules(args.data_dir, args.dataset)
    except:  # noqa: E722
        raise NotImplementedError(f"unrecognized dataset '{args.dataset}'")

    # Create Results folder
    args.res_dir = f"./results/{args.dataset}"
    os.makedirs(args.res_dir, exist_ok=True)

    # Create folder to save log file and response objects
    args.expt_data_dir = (
        f"./objects_and_logs/{system_name}_{args.model}_{args.embedder}"
    )
    os.makedirs(args.expt_data_dir, exist_ok=True)

    print("Created dataset")
    item_list = ds._get_data()

    human = log.HumanLogger(args.expt_data_dir)
    with human.for_id(f"trace_{dataset_name}") as hlog:
        if system_name == "random":
            response_list = []
            system = methods.Random(args.dataset)
            for i in range(args.n_runs):
                rd_results = system.get_preds(
                    hlog, item_list, args.n_rounds, args.n_sample_per_round
                )
                response_list.append(rd_results)
                hlog.write(f"------------Run {i+1} finished.\n")
            save_info_about_experiment(response_list, args)

        elif system_name == "coreset":
            response_list = []
            system = methods.Coreset(args.dataset, args.embedder)
            for i in range(args.n_runs):
                rd_results = system.get_preds(
                    hlog, item_list, args.n_rounds, args.n_sample_per_round
                )
                response_list.append(rd_results)
                hlog.write(f"------------Run {i+1} finished.\n")
            save_info_about_experiment(response_list, args)

        elif system_name == "gp":
            response_list = []
            system = methods.GPR(args.dataset, args.embedder)
            for i in range(args.n_runs):
                rd_wise_perf = system.get_preds(
                    hlog, item_list, args.n_rounds, args.n_sample_per_round
                )
                response_list.append(rd_wise_perf)
                hlog.write(f"------------Run {i+1} finished.\n")

            save_info_about_experiment(response_list, args)

        elif system_name == "adaptive_gp":
            response_list = []
            system = methods.AdaptiveGPR(args.dataset, args.embedder)
            for i in range(args.n_runs):
                try:
                    rd_wise_perf = system.get_preds(
                        hlog, item_list, args.n_rounds, args.n_sample_per_round
                    )
                    response_list.append(rd_wise_perf)
                    hlog.write(f"------------Run {i+1} finished.\n")
                except:  # noqa: E722
                    continue
            save_info_about_experiment(response_list, args)

        elif system_name == "llmnn":
            logger.info("creating store...")
            store = None
            port = random.randint(1000, 65534)
            with NiceWeaviate(port, port + 1) as client:
                logger.info(f"{system_name}: Creating collection and uploading data")
                store = retrieval.DataStore(
                    client, "Data", "Candidate Data for BO", args.embedder, dataset_name
                )
                store.populate(logger, item_list)

                model = Model.from_full_name(
                    args.model, model_path=args.models_dir, temp=args.temp, rank=0
                )

                print("Created model")
                response_list = []
                for i in range(args.n_runs):
                    strategy = methods.LLMNN(model, store, dataset_name)
                    sys_resp = strategy.generate(
                        hlog, args.n_rounds, args.n_sample_per_round, args.n_centroids
                    )
                    response_list.append(sys_resp)
                    store.refresh_store()  # Mark every candidate as unexplored for the new run
                    print(f"Finished loop {i+1}")
                    hlog.write(f"------------Run {i+1} finished.\n")

                save_info_about_experiment(response_list, args)
            try:
                pass
            finally:
                if store is not None:
                    store.embedder.flush()
                client.close()
