from typing import List

from metaflow import FlowSpec, step, pip, batch, project, conda_base, Parameter, current
import numpy.typing as npt

try:  # Hack for type-hints on attributes
    from pandas import DataFrame
except ImportError:
    pass


ENCODER_MODEL = "multi-qa-MiniLM-L6-cos-v1"


@conda_base(python="3.9")
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

        self.org_descriptions = org_descriptions.loc[self.org_ids]

        if self.test_mode and not current.is_production:
            self.org_descriptions = self.org_descriptions.head(1000)

        self.next(self.embed_descriptions)

    @batch(memory=32_000)
    @pip(path="requirements.txt")
    @step
    def embed_descriptions(self):
        """Apply transformer to Glass descriptions"""
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(ENCODER_MODEL)
        self.embeddings = encoder.encode(self.org_descriptions["description"].to_list())

        del self.org_descriptions

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    GlassEmbed()
