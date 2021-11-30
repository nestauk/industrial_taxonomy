# Steps in data pipeline

## 1. Collect data

### Official data

Run `python industrial_taxonomy/pipeline/sic/flow.py --production run` to create the SIC code - name lookups

Run `python industrial_taxonomy/pipeline/official/population/flow.py --production run` to collect population data

Run `python industrial_taxonomy/pipeline/official/gdp/flow.py --production run` to collect GDP data

Run `python industrial_taxonomy/pipeline/official/nomis/flow.py --production run` to collect Nomis data
