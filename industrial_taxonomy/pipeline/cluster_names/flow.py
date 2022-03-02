import logging
from metaflow import (
    conda,
    current,
    FlowSpec,
    project,
    step,
    Parameter,
)

from industrial_taxonomy.getters.glass_house import (
    description_embeddings,
    embedded_org_descriptions,
    embedded_org_ids,
    encoder_name,
)

from utils import (
    split_dict,
    sector_org_ids_lookup,
    get_clusters_embeddings,
    tfidf_vectors,
    top_tfidf_terms,
    central_ngrams,
)


logger = logging.getLogger(__name__)


@project(name="industrial_taxonomy")
class TextSectorName(FlowSpec):

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @conda(
        libraries={
            "sentence-transformers": "2.2.0",
        },
    )
    @step
    def start(self):
        """Instantiates the sentence transformer used to embed Glass + CH
        descriptions."""
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(encoder_name())
        self.next(self.load_text_sectors)

    @conda(
        libraries={
            "graph-tool": "2.44",
        },
    )
    @step
    def load_text_sectors(self):
        """Loads text sectors and truncates if in test mode."""
        from industrial_taxonomy.getters.text_sectors import (
            assigned_text_sector,
            org_ids_reassigned,
        )

        text_sectors = assigned_text_sector()
        org_ids = org_ids_reassigned()
        min_len = min([len(c) for c in text_sectors.values()])
        test_params = [k for k, v in text_sectors.items() if len(v) == min_len]

        if self.test_mode and not current.is_production:
            params = test_params
        else:
            params = text_sectors.keys()

        self.params, text_sectors = split_dict(text_sectors, params)
        _, org_ids = split_dict(org_ids, params)

        self.text_sectors = []
        for t, o in zip(text_sectors, org_ids):
            self.text_sectors.append(list(zip(t, o)))

        self.next(self.name_text_sectors, foreach="text_sectors")

    @conda(
        libraries={
            "scikit-learn": "1.0.2",
            "sentence-transformers": "2.2.0",
        }
    )
    @step
    def name_text_sectors(self):
        """Generates names for the text sectors."""

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import pairwise_distances

        org_ids = embedded_org_ids()
        embeddings = description_embeddings()
        descriptions = embedded_org_descriptions()

        sector_org_ids_index = sector_org_ids_lookup(self.input)

        self.sector_names = {}

        for label, sector_org_ids in sector_org_ids_index.items():
            if len(sector_org_ids) <= 1:
                self.sector_names[label] = None
                continue

            sector_clusters = [(label, org_id) for org_id in sector_org_ids]
            sector_embeddings, sector_locs = get_clusters_embeddings(
                sector_clusters,
                org_ids,
                embeddings,
            )
            sector_descriptions = [descriptions[i] for i in sector_locs]

            if len(sector_descriptions) > 2:
                min_df = 2
            else:
                min_df = 1

            try:
                sector_tfidf_vecs, tfidf_vectorizer = tfidf_vectors(
                    sector_descriptions,
                    TfidfVectorizer,
                    min_df=min_df,
                )
            # some text sectors contain too few terms to calculate tf-idf
            except ValueError:
                self.sector_names[label] = None
                continue

            sector_top_ngrams = top_tfidf_terms(
                sector_tfidf_vecs,
                tfidf_vectorizer,
                topn=20,
            )

            best_ngrams = central_ngrams(
                sector_top_ngrams,
                self.encoder,
                sector_embeddings,
                pairwise_distances,
                topn=3,
            )
            self.sector_names[label] = ", ".join(best_ngrams)

        self.next(self.join)

    @step
    def join(self, inputs):
        """Joins names back into dict with clustering params."""
        sector_names = [input.sector_names for input in inputs]
        self.merge_artifacts(inputs, exclude=["sector_names"])
        self.sector_names = dict([(p, s) for p, s in zip(self.params, sector_names)])

        self.next(self.end)

    @step
    def end(self):
        """No-op"""
        pass


if __name__ == "__main__":
    TextSectorName()
