# Methodology

Here we describe the prototype methodology that we developed for our previous report, some of the limitations that we identified and the strategies we have implemented to address them, and outstanding issues.

## Prototype methodology and its limitations

The methodology for our prototype taxonomy consisted of the following steps (see Bishop et al (2022) for additional detail).

1. Match the company database obtained from Glass, our data provider, with Companies House in order to label glass companies with their SIC code
2. Pre-process the text in company descriptions from the glass data.
3. Cluster companies into `text sectors` locally (inside their 4-digit SIC code) using topsbm, a topic modelling algorithm that clusters documents based on the similarity between their text content.
4. Reassign companies globally (across all text sectors generated in step 3) based on their semantic similarity to a "text cluster" summary. This is performed iteratively until the reassignment procedure stabilises.
5. Build a network of text sectors based on their co-occurrence in companies and decompose it hierarchically in order to build a hierarchical industrial taxonomy.  
6. Build text sector corpora that concatenate the descriptions of all the companies in them and identify salient terms (based on Term-Frequency Inverse-Document-Frequency - TF-IDF) which are used to name them.
7. Analyse the outputs

This taxonomy presented some important limitations:

* Step 3 was performed on 4-digit SIC codes with more than 2,000 companies in the Glass data, limiting our coverage to 43 SIC codes.
* Step 3 assigned _all_ companies in a sector to a text sector, in effect forcing edge cases into text sectors in a way that might make them noisy.
* Step 4 assigned each company to its closest sector but did not provide a measure of the confidence for this assignment based on, for example, its distance to its closest neighbouring sector, or the heterogeneity of the text sectors in its vicinity.
* Step 6 calculated a text sector's TF-IDF compared to _all other_ text sectors at the lowest level of the taxonomy, resulting in text sectors with uninformative names (e.g. all companies in text sectors related to finance were labelled with similar names).

## Upgrades

We have sought to address the issues above by implementing the following changes in the pipeline.

1. We have lowered the SIC threshold for inclusion into the pipeline to 1,000 companies, increasing our coverage to 106 4-digit SIC codes.
2. We have performed the text clustering with different parameters regulating the inclusion of companies into text sectors. Lower values for this parameter mean that fewer companies - those that are most representative - are assigned to a cluster, potentially reducing noise in the seed text clusters that are used in the reassignment procedure in Step 4. At the same time, this means that the representation of text sectors in step 3 is based on fewer observations (for example, 10 vs. more than 1000), which might make them less stable. Since it is unclear, ex-ante, which strategy will perform better, we have implemented a "consequential evaluation" procedure based on the empirical correlation between taxonomy outputs and secondary data to select an assignment strategy for downstream analysis (see results in Section 3).
3. ...
## Outstanding issues

...