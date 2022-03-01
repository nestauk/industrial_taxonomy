from typing import List

from metaflow import FlowSpec, step, pip, batch, project, Parameter, current
import numpy.typing as npt

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass

MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


@project(name="industrial_taxonomy")
class GlassEmbed(FlowSpec):
    """Transform descriptions of fuzzy matched companies into embeddings.

    This uses the multi-qa-mpnet-base-dot-v1 transformer model which encodes up
    to 512 tokens per document and produces embeddings with 768 dimensions.
    It is recommended by SBERT due to its performance when benchmarked against
    other transformers for semantic search:
    https://www.sbert.net/docs/pretrained_models.html

    The model produces normalised embeddings of length 1, meaning that the dot
    and cosine products are equivalent.

    Attributes:
        embeddings: Embeddings of Glass org descriptions
        model_name: Name of pre-trained transformer model used to generate encodings
        org_descriptions: Descriptions of Glass organisations that are embedded
        org_ids: Glass IDs for orgs with embeddings (follows order of embeddings)
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    embeddings: npt.ArrayLike
    model_name: str
    org_descriptions: "DataFrame"
    org_ids: List

    @step
    def start(self):
        """Load matched Glass and Companies House IDs and split into chunks
        for embedding.
        """

        from industrial_taxonomy.getters.glass import get_organisation_description
        from industrial_taxonomy.getters.glass_house import glass_companies_house_lookup

        org_descriptions = get_organisation_description()
        self.org_ids = org_descriptions.index.intersection(
            list(glass_companies_house_lookup().keys())
        ).to_list()

        nrows = 50_000 if self.test_mode and not current.is_production else None
        self.org_ids = self.org_ids[:nrows]

        self.org_descriptions = org_descriptions.loc[self.org_ids][
            "description"
        ].to_list()

        self.next(self.embed_descriptions)

    @batch(
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        # Queue gives p3.2xlarge, with:
        gpu=1,
        memory=60000,
        cpu=8,
    )
    @pip(libraries={"sentence-transformers": "2.1.0"})
    @step
    def embed_descriptions(self):
        """Apply transformer to Glass descriptions"""
        from sentence_transformers import SentenceTransformer
        from torch import cuda

        if not cuda.is_available():
            raise EnvironmentError("CUDA is not available")

        self.model_name = MODEL_NAME
        encoder = SentenceTransformer(self.model_name)
        self.embeddings = encoder.encode(self.org_descriptions)

        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    GlassEmbed()
