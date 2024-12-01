from mp_api.client import MPRester
import numpy as np
import pandas as pd

allowed_elem_list = ["H", "C", "N", "O"]


def check_elements(element_list):
    for elem in element_list:
        if elem.name not in allowed_elem_list:
            return False
    return True


smile_list = []
ie_list = []
with MPRester("IsWIXZ2E1zhahE1sriTXsM6jpqW3sePq") as mpr:
    docs = mpr.molecules.jcesr._search(
        IE_min=-np.inf, fields=["IE", "EA", "smiles", "elements"]
    )

    for doc in docs:
        if doc.IE is not None:
            if check_elements(doc.elements) and doc.IE > -10 and doc.IE < 10:
                smile_list.append(doc.smiles)
                ie_list.append(round(doc.IE, 3))


df = pd.DataFrame({"SMILES": smile_list, "IE": ie_list})
df.to_csv("./nextgen_dataset.csv", encoding="utf-8", index=False)
