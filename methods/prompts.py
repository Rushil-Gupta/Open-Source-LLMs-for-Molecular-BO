_sys_prompt = (
    "You are a helpful chemistry expert. Your role is to help me in accomplishing the task I "
    "describe to you."
)

_human_prompt = (
    "[TASK]\n"
    "Out of a set of {num_candidates} molecules, my goal is to "
    "identify the maximum number of molecules with highest {score_description}. "
    "I am allowed to conduct wet lab experiments in batches of {batch_size} over {n_rounds} rounds."
    " After every round of experiment, I will provide you with feedback on your predictions, "
    "including the correctly identified "
    "molecules called hits and the property value called as the score. {additional_info}\n"
    "[END TASK]\n"
    "This is round {rd}. Here is the feedback of your prediction until now:\n"
    "{feedback}\n\n"
    "Please propose {n_centroids} different yet valid SMILES strings that you want to explore "
    "next. Note that I will choose unexplored molecules closest to your predicted SMILES strings to"
    " form the predictions. Your response should exactly be a valid JSON in the following format:\n"
    '{{\n"Rationale": "<Explanation of your choices>",\n"SMILES": [<List of SMILES strings>]\n}}'
)
