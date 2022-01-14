from operator import itemgetter
from typing import List

from metaflow import FlowSpec, step, pip, project, Parameter, current, conda

from industrial_taxonomy.getters.glass_house import (
    description_embeddings,
    embedded_org_descriptions,
    encoder_name,
    embedded_org_ids,
)
from qa_utils import (
    token_count,
    tokenized_length_histogram,
    random_sample_idx,
    get_latest_runs_by_model,
)


@project(name="industrial_taxonomy")
class GlassEmbedQA(FlowSpec):
    """Produces outputs for QA of results from GlassEmbed flow.

    Attributes:
        tokenized_length_hist_fig: Cumulative histogram of tokenized
            Glass description lengths.
        percent_not_truncated: Percentile ranking of the maximum
            number of input tokens for transformer with respect to the
            distrubtion of tokenixed Glass description lengths.
        embedding_matches: A list of dictionaries containing company pairs
            matched by embeddings of their descriptions using a FAISS index.
            This is for manual evaluation of company description similarity.
    """

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @step
    def start(self):
        """Set test size if in test mode."""
        self.nrows = 10_000 if self.test_mode and not current.is_production else None
        self.next(self.truncated_descriptions)

    @conda(
        libraries={
            "transformers": "4.15.0",
            "scipy": "1.7.3",
        }
    )
    @step
    def truncated_descriptions(self):
        """Calculates and plots the normalised cumulative ranking of the
        model's maximum input length. This indicates the fraction of
        Glass descriptions which are truncated when embeddings are
        calculated."""
        from scipy.stats import percentileofscore
        from transformers import AutoTokenizer

        descriptions = embedded_org_descriptions()
        sample_idx = random_sample_idx(descriptions, 100_000)
        descriptions = list(itemgetter(*sample_idx)(descriptions))[: self.nrows]

        model_name = encoder_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        description_lengths = token_count(descriptions, tokenizer)

        self.tokenized_length_hist_fig = tokenized_length_histogram(
            description_lengths,
            tokenizer,
        )
        self.percent_not_truncated = percentileofscore(
            description_lengths, tokenizer.model_max_length
        )

        self.next(self.embedding_neighbours)

    @pip(libraries={"faiss-cpu": "1.7.2"})
    @step
    def embedding_neighbours(self):
        """Generates a list of randomly sampled Glass IDs and their
        descriptions along with their nearest neighbours found using
        their embeddings.
        """
        import faiss

        org_ids = embedded_org_ids()[: self.nrows]
        descriptions = embedded_org_descriptions()[: self.nrows]
        sample_idx = random_sample_idx(org_ids, 100)

        model_runs = get_latest_runs_by_model()

        self.embedding_matches = {
            org_ids[sid]: {"sample_description": descriptions[sid]}
            for sid in sample_idx
        }
        for run in model_runs:

            embeddings = description_embeddings(run)[: self.nrows]
            model_name = encoder_name(run)

            dimensions = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimensions)
            index.add(embeddings)

            distances, nearest_neighbours_idx = index.search(embeddings[sample_idx], 2)
            distances = distances[:, 1]  # nearest neighbours that aren't self
            nearest_neighbours_idx = nearest_neighbours_idx[:, 1]

        for sample_id, nearest_id, dist in zip(
            sample_idx, nearest_neighbours_idx, distances
        ):
            org_id = org_ids[sample_id]
            self.embedding_matches[org_id][model_name] = {
                "nearest_glass_org_id": org_ids[nearest_id],
                "sample_glass_description": descriptions[sample_id],
                "nearest_glass_description": descriptions[nearest_id],
                "distance": dist,
            }
        self.next(self.end)

    @step
    def end(self):
        """No-op."""
        pass


if __name__ == "__main__":
    GlassEmbedQA()
