import matplotlib.pyplot as plt
import numpy as np
import os

gray = "#c8c8c8"
gold = "#f0c571"
teal = "#59a89c"
blue = "#0b81a2"
red = "#e25759"
darkred = "#9d2c00"
purple = "#7e4794"
green = "#36b700"

color_list = [red, green, purple, gold, blue, darkred, teal]

label_dict = {
    "random": "Random",
    "coreset_molformer": "CoreSet",
    "gp_molformer": "GPR",
    "llmnn_molformer_hf-llama3.1-8b-instruct": "LLMNN-Llama",
    "llmnn_molformer_hf-qwen2-7b-instruct": "LLMNN-Qwen",
}


datasets = ["nextgen"]
llms = ["hf-llama3.1-8b-instruct", "hf-qwen2-7b-instruct"]
methods = ["random", "coreset_molformer", "gp_molformer"]
methods.extend([f"llmnn_molformer_{llm}" for llm in llms])

path_dir = "./plots"
os.makedirs(path_dir, exist_ok=True)

x_vals = [1, 2, 3, 4, 5]
for j, data in enumerate(datasets):
    fname = f"all_molformer_{data}"
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, meth in enumerate(methods):
        try:
            with open(f"./results/{data}/{meth}.txt", "r") as f:
                file_str = f.readline()
                rd_wise_frac = [float(mean_str) for mean_str in file_str[1:-2].split()]
                rd_wise_frac = np.cumsum(np.round(np.array(rd_wise_frac), 2)).tolist()
            ax.plot(
                x_vals,
                rd_wise_frac,
                color=color_list[k],
                label=label_dict[meth],
                marker="o",
                markersize=7,
                linewidth=2.5,
            )

        except:  # noqa: E722
            continue

    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis="both", labelsize=20)
    plt.rcParams["ytick.labelsize"] = 20
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.legend()
    ax.set_ylabel("Cumulative hits", fontsize=20)
    ax.set_xlabel("Round", fontsize=20)
    ax.set_xticks(1 + np.arange(5), x_vals)
    plt.legend(fontsize=16)
    plt.grid(which="both")
    plt.savefig(f"{path_dir}/{fname}.png", bbox_inches="tight")  # Save the figure
    plt.savefig(
        f"{path_dir}/{fname}.pdf", format="pdf", bbox_inches="tight"
    )  # Save the figure
    plt.close()
