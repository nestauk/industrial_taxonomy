# Steps in data pipeline

:warning: When testing omit the `--production` flag from flow commands

## 1. Collect data

### Official data

**SIC code - Name lookups**

```bash
python industrial_taxonomy/pipeline/sic/flow.py \
--production \
run
```

**Population data**

```bash
python industrial_taxonomy/pipeline/sic/flow.py \
--production \
run
```

**GDP data**

```bash
python industrial_taxonomy/pipeline/official/population/flow.py \
--production \
run
```

**Nomis data**

```bash
python industrial_taxonomy/pipeline/official/nomis/flow.py \
--production \
run
```

**Other local benchmarking data**

```bash
python industrial_taxonomy/pipeline/official/local_benchmark/flow.py \
--production \
run
```

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

## 4. Cluster glass descriptions into text sectors

```bash
python industrial_taxonomy/pipeline/glass_clusters/flow.py \
--datastore=s3 \
--production  run \
```

## 5. Vectorise Glass descriptions

```bash
python industrial_taxonomy/pipeline/glass_embed/flow.py \
 --production \
 --no-pylint \
 run
```

## 6. Reassign clusters

```bash
python industrial_taxonomy/pipeline/cluster_reassign/flow.py \
 --production \
 --environment=conda \
 --no-pylint \
 run
```
