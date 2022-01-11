from typing import List

from metaflow import FlowSpec, step, pip, project, Parameter, current, conda, conda_base

try:
    from pandas import DataFrame
except ImportError:
    pass

from industrial_taxonomy.getters.glass_house import (
    embedded_org_descriptions,
    encoder_name,
)
from industrial_taxonomy.pipeline.glass_embed.qa_utils import (
    full_text_tokenize,
    tokenized_length_histogram,
    tokenized_original_length,
)


@project(name="industrial_taxonomy")
class GlassEmbedQA(FlowSpec):
    """Produces outputs for QA of results from GlassEmbed flow."""

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @step
    def start(self):
        self.next(self.check_truncated_descriptions)

    @conda(libraries={"transformers": "4.15.0", "scipy": "1.7.3"})
    @step
    def check_truncated_descriptions(self):
        """Calculates"""
        from scipy.stats import percentileofscore
        from transformers import AutoTokenizer

        descriptions = embedded_org_descriptions()
        model_name = encoder_name()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenized = full_text_tokenize(descriptions, tokenizer)
        description_lengths = tokenized_original_length(tokenized)
        self.tokenized_length_histogram_fig = tokenized_length_histogram(
            description_lengths,
            tokenizer,
        )
        self.percentile_model_max_length = percentileofscore(description_lengths)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    GlassEmbedQA()
