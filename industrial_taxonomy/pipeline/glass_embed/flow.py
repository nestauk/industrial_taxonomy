from typing import List

from metaflow import FlowSpec, step, pip, batch, project, Parameter, current
import numpy as np
import numpy.typing as npt

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass

from industrial_taxonomy.pipeline.glass_embed.utils import chunks

ENCODER_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


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
        ).to_list()

        org_descriptions = org_descriptions.loc[self.org_ids]

        if self.test_mode and not current.is_production:
            test_size = 200_000
            org_descriptions = org_descriptions.head(test_size)
            self.org_ids = self.org_ids[:test_size]

        org_descriptions = org_descriptions["description"].to_list()

        batch_size = 100_000
        self.org_description_chunks = chunks(org_descriptions, batch_size)

        self.next(self.embed_descriptions, foreach="org_description_chunks")

    @batch(
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        # Queue gives p3.2xlarge, with:
        gpu=1,
        memory=61000,
        cpu=8,
    )
    @pip(path="requirements.txt")
    @step
    def embed_descriptions(self):
        """Apply transformer to Glass descriptions"""
        from sentence_transformers import SentenceTransformer
        from torch import cuda

        if not cuda.is_available():
            raise EnvironmentError("CUDA is not available")

        encoder = SentenceTransformer(ENCODER_MODEL)
        self.embeddings_chunk = encoder.encode(self.input)

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
