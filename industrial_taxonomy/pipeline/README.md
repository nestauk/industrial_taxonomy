# Steps in data pipeline

:warning: When testing omit the `--production` flag from flow commands

## 1. Collect data

### Official data

Run `python industrial_taxonomy/pipeline/sic/flow.py --production run` to create the SIC code - name lookups

Run `python industrial_taxonomy/pipeline/official/population/flow.py --production run` to collect population data

Run `python industrial_taxonomy/pipeline/official/gdp/flow.py --production run` to collect GDP data

Run `python industrial_taxonomy/pipeline/official/nomis/flow.py --production run` to collect Nomis data

## 2. Fuzzy-match Glass organisations to Companies House companies

```bash
python industrial_taxonomy/pipeline/glass_house/flow.py \
 --environment=conda \
 --production \
 run \
 --clean-names true \
 --with batch
```

## 3. Tokenise and N-gram Glass descriptions

```bash
python industrial_taxonomy/pipeline/glass_description_ngrams/nlp_flow.py \
 --environment=conda \
 --production \
 run \
 --n-gram 3 \
 --n-process 2 \
 --with batch:memory=32000,cpu=2
```
