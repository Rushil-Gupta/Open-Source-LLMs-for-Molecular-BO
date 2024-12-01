import logging
import numpy as np
from weaviate import WeaviateClient
import weaviate.classes.config as wc
import os
from dataset import Candidate

from .store import Store


class DataStore(Store):
    def __init__(
        self, store: WeaviateClient, name: str, desc: str, embedder: str, dataset: str
    ):
        super().__init__(store, name, desc, embedder, dataset)

    @property
    def coll_properties(self) -> list[wc.Property]:
        _coll_properties = [
            wc.Property(
                name="name",
                data_type=wc.DataType.TEXT,
                description="The identifier of each candidate",
            ),
            wc.Property(
                name="score",
                data_type=wc.DataType.NUMBER,
                description="The score for each candidate",
            ),
            wc.Property(
                name="explored",
                data_type=wc.DataType.BOOL,
                description="Whether this candidate has already been explored",
            ),
        ]
        return _coll_properties

    def populate(
        self, logger: logging.Logger, mols: list[Candidate], batch_size: int = 4096
    ):
        """Vectorize, cache, and upload the candidates."""
        logger.debug("embedding %d candidates", len(mols))
        s_id = 0
        b_id = 0
        save_dir = f"./cache/{self.dataset}/{self.embedder.e.model_str}"
        os.makedirs(save_dir, exist_ok=True)
        while s_id < len(mols):
            e_id = min(s_id + batch_size, len(mols))
            batch_mols = mols[s_id:e_id]

            if os.path.exists(f"{save_dir}/embeds_batch{b_id}.npy"):
                vectors = np.load(
                    f"{save_dir}/embeds_batch{b_id}.npy", allow_pickle=True
                )
            else:
                text_to_embed = [m.name for m in batch_mols]
                vectors = self.embedder.embed(text_to_embed)

                np.save(
                    f"{save_dir}/embeds_batch{b_id}.npy",
                    np.array(vectors, dtype=object),
                    allow_pickle=True,
                )

            s_id += batch_size
            b_id += 1
            self.weaviate_insert(
                logger, [p.to_dict(self.dataset) for p in batch_mols], vectors
            )
            print(f"Finished adding {e_id} molecules")

    def return_all(self) -> list[Candidate]:
        out = []

        for obj in self.collection.iterator():
            out.append(Candidate(obj.properties["name"], obj.properties["score"]))
        return out

    def get_unexplored_objects(self, ret_vectors=False):
        out = []
        vector_list = []
        for obj in self.collection.iterator(include_vector=True):
            if not obj.properties["explored"]:
                out.append(
                    (
                        obj.uuid,
                        Candidate(obj.properties["name"], obj.properties["score"]),
                    )
                )
                vector_list.append(obj.vector["default"])
        if ret_vectors:
            return out, vector_list
        else:
            return out

    def change_element_status_by_uuid(self, uuid, explored):
        self.collection.data.update(
            uuid=uuid,
            properties={
                "explored": explored,
            },
        )

    def update_objects_and_get_expt_results(self, preds: list[str]):
        ret_list = []
        uuid_list = []
        for obj in self.collection.iterator():
            if obj.properties["name"] in preds:
                ret_list.append(
                    {
                        "name": obj.properties["name"],
                        "score": round(obj.properties["score"], 2),
                    }
                )

                uuid_list.append(obj.uuid)
                self.collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "explored": True,
                    },
                )

        return ret_list, uuid_list

    def update_objects_as_explored(self, preds: list[str]):
        for obj in self.collection.iterator():
            if obj.properties["name"] in preds:
                self.collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "explored": True,
                    },
                )

    def refresh_store(self):
        for obj in self.collection.iterator():
            uuid = obj.uuid
            self.collection.data.update(
                uuid=uuid,
                properties={
                    "explored": False,
                },
            )

    def search(self, query: str, k: int, rd: int) -> list[str]:
        """Returns the nearest candidates to the chosen centroid"""
        try:
            embedded_query = self.embedder.embed([query], is_query=True)[0]
            out = []
            expt_result = []
            obj_uuids = []
            count = 0
            factor = rd
            while count < k:
                res = self.collection.query.near_vector(
                    near_vector=embedded_query.tolist(),
                    limit=factor * k,
                    return_properties=["name", "score", "explored"],
                )

                for obj in res.objects:
                    if not obj.properties["explored"]:
                        out.append(obj.properties["name"])
                        expt_result.append(
                            {
                                "name": obj.properties["name"],
                                "score": round(obj.properties["score"], 2),
                            }
                        )
                        count += 1

                        self.collection.data.update(
                            uuid=obj.uuid, properties={"explored": True}
                        )
                        obj_uuids.append(obj.uuid)

                    if count == k:
                        break

                factor = 2 * factor
                if count != k:
                    print(f"Searching again with factor {factor}")

            if len(out) != k:
                raise ValueError(
                    f"Asked {k} candidates. However, got only {len(out)} candidates in search."
                )

            return out, expt_result, obj_uuids
        except:  # noqa: E722
            return None, None, None
