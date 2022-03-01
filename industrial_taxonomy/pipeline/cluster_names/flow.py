import logging
import numpy as np

from metaflow import (
    conda,
    current,
    FlowSpec,
    project,
    step,
    batch,
    Parameter,
)

logger = logging.getLogger(__name__)

from industrial_taxonomy.getters.glass_house import (
    description_embeddings,
    embedded_org_descriptions,
    embedded_org_ids,
)


@project(name="industrial_taxonomy")
class TextSectorName(FlowSpec):

    test_mode = Parameter(
        "test-mode",
        help="Whether to run in test mode (on a small subset of data)",
        type=bool,
        default=lambda _: not current.is_production,
    )

    @step
    def start(self):
        pass

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TextSectorName()
