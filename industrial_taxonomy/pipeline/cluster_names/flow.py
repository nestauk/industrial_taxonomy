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
    sic4_lookups,
    get_locs,
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
        sic4_org_id_lookup, sic4_text_sector_lookup = sic4_lookups(
            sector_org_ids_index,
        )

        self.sector_names = {}

        for sic4, text_sectors in sic4_text_sector_lookup.items():
            sic4_org_ids = sic4_org_id_lookup[sic4]
            sic4_clusters = [(sic4, org_id) for org_id in sic4_org_ids]

            _, sic4_locs = get_clusters_embeddings(
                sic4_clusters,
                org_ids,
                embeddings,
            )
            sic4_descriptions = [descriptions[i] for i in sic4_locs]

            sic4_tfidf_vecs, tfidf_vectorizer = tfidf_vectors(
                sic4_descriptions,
                TfidfVectorizer,
                min_df=2,
                ngram_range=(2, 3),
            )

            for text_sector in text_sectors:
                sector_org_ids = sector_org_ids_index[text_sector]
                n_companies = len(sector_org_ids)

                logger.info(f"Generating names for {text_sector}")
                logger.info(f"Number of companies: {n_companies}")

                if n_companies < 2:
                    self.sector_names[text_sector] = None
                    continue

                sector_clusters = [(text_sector, org_id) for org_id in sector_org_ids]
                sector_embeddings, _ = get_clusters_embeddings(
                    sector_clusters,
                    org_ids,
                    embeddings,
                )
                tfidf_sector_locs = get_locs(sector_org_ids, sic4_org_ids)

                sector_top_ngrams = top_tfidf_terms(
                    sic4_tfidf_vecs[tfidf_sector_locs],
                    tfidf_vectorizer,
                    topn=20,
                )

                logger.info(f"Found {len(sector_top_ngrams)} term candidates")
                if len(sector_top_ngrams) < 2:
                    self.sector_names[text_sector] = ", ".join(sector_top_ngrams)
                    continue

                name_length = 5
                best_ngrams = central_ngrams(
                    sector_top_ngrams,
                    self.encoder,
                    sector_embeddings,
                    pairwise_distances,
                    topn=name_length,
                )

                self.sector_names[text_sector] = ", ".join(best_ngrams)

        self.next(self.join)

    @step
    def join(self, inputs):
        """Joins names back into dict with clustering params."""
        sector_names = [input.sector_names for input in inputs]
        self.merge_artifacts(inputs, exclude=["sector_names", "encoder"])
        self.sector_names = dict([(p, s) for p, s in zip(self.params, sector_names)])

        self.next(self.end)

    @step
    def end(self):
        """No-op"""
        pass


if __name__ == "__main__":
    TextSectorName()
