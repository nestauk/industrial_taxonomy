from typing import List

from metaflow import FlowSpec, step, pip, batch, project, conda_base, Parameter, current
import numpy as np
import numpy.typing as npt

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass

ENCODER_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


@conda_base(python="3.8")
@project(name="industrial_taxonomy")
class GlassEmbed(FlowSpec):
    """Transform descriptions of fuzzy matched companies into embeddings.

    Attributes:
        org_descriptions: Descriptions that are encoded
        model: Pre-trained transformer model used to perform encodings
        embeddings: Embeddings of org_descriptions
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    org_descriptions: "DataFrame"
    org_ids: List
    model: str
    embeddings: npt.ArrayLike

    @step
    def start(self):
        """Load matched Glass and Companies House IDs"""

        from industrial_taxonomy.getters.glass import get_organisation_description
        from industrial_taxonomy.getters.glass_house import glass_companies_house_lookup

        org_descriptions = get_organisation_description()
        self.org_ids = org_descriptions.index.intersection(
            list(glass_companies_house_lookup().keys())
        )
        org_descriptions = org_descriptions.loc[self.org_ids]

        if self.test_mode and not current.is_production:
            org_descriptions = org_descriptions.head(200_000)

        batch_size = 100_000
        # self.org_description_chunks = chunks(
        #     org_descriptions, batch_size
        # )
        self.org_description_chunks = [
            org_descriptions[i : i + batch_size]
            for i in range(0, len(org_descriptions), batch_size)
        ]

        self.next(self.embed_descriptions, foreach="org_description_chunks")

    @batch(gpu=1, queue="job-queue-GPU-nesta-metaflow", image="pytorch/pytorch")
    @pip(path="requirements.txt")
    @step
    def embed_descriptions(self):
        """Apply transformer to Glass descriptions"""
        # from industrial_taxonomy.pipeline.glass_embed.utils import encode, chunks, load_model, load_tokenizer
        from sentence_transformers import SentenceTransformer
        from torch import cuda

        assert cuda.is_available(), "Cuda not available!"
        # tokenizer = load_tokenizer(ENCODER_MODEL)
        # model = load_model(ENCODER_MODEL)

        # chunk_size = 100
        # embedding_chunks = []
        # for chunk in chunks(self.input['description'].to_list(), chunk_size):
        #     embedding_chunks.append(
        #         encode(chunk, tokenizer, model)
        #     )

        # self.embeddings_chunk = np.concatenate(embedding_chunks)

        encoder = SentenceTransformer(ENCODER_MODEL)
        self.embeddings_chunk = encoder.encode(self.input["description"].to_list())

        self.next(self.join)

    @step
    def join(self, inputs):
        self.embeddings = np.concatenate([input.embeddings_chunk for input in inputs])

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    GlassEmbed()
