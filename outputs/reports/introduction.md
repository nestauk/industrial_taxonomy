# Introduction

Many policy agendas require access to high resolution, timely information about the industrial composition of the economy. For example, the UK Government's "Levelling Up" agenda has the goal of reducing economic and social disparities across the country. The industrial composition of a location is both a determinant and a consequence of those disparities, so being able to measure it accurately can help identify challenges and opportunities for levelling up. On a similar vein, policymakers are interested in nurturing innovative, emerging industries such as AI, and sectors that can support and accelerate the transition to a net-zero economy. Accurate information about the levels of activity in those industries, as well as their evolution and geography is required to design, target, monitor and evaluate those policies.

The official taxonomies that are generally used to measure the industrial composition of the economy present some important limitations which reduces their usefulness for these policy use cases: they miss new industries that have emerged after the structure of the taxonomy was agreed; they suffer from misclassification and uninformative categories containing heterogeneous collections of sectors; they are less suitable for companies that straddle sectors. 

In a previous ESCoE paper, we used business website data and state-of-the-art machine learning methods to evidence these limitations with a focus on the 2007 Standard Industrial Classification (SIC) taxonomy in use in the UK. We also developed an alternative 'bottom-up' taxonomy based on the text in company descriptions that addresses some of those limitations. Our analysis revealed high levels of heterogeneity in the most knowledge-intensive 4 digit SIC codes and important overlaps between the text descriptions of companies in different parts of the taxonomy suggesting that fuzzy links between SIC categories and company activities could be leading to misclassification and noise in industrial statistics (particularly at higher levels of sectoral resolution). An exploratory analysis of our prototype taxonomy, which clustered companies into "text sectors" based on similarities between their web descriptions, suggested that it could usefully augment official taxonomies by e.g. 

* helping decompose heterogeneous 4-digit SIC codes into more finaly grained sectors, 
* making it possible to study policy-relevant industries currently absent from the SIC taxonomy such as "the green economy", and 
* characterise more accurately the composition of local economies.

For this paper, we have upgraded and enhanced the pipeline underpinning the taxonomy and analysed the outputs to address research questions related to the Levelling Up policy agenda.

The structure of the paper is as follows: 

Section 2 summarises the methodology for the original taxonomy and its limitations, and describes the strategy that we have adopted to address those limitations in this new implementation. 

Section 3 outlines the secondary data we have relied on, and how we have used it to "consequentially evaluate" the performance of versions of the taxonomy based on different parameters, and select the best-performing one which we use in downstream analysis.

Section 4 shows how we have used co-occurrences between sectors in companies, one of the outputs from our pipeline, to reconstruct and visualise a hierarchical taxonomy that can be examined at different levels of aggregation.

Section 5 presents the findings of an analysis of the industrial geography of the UK based on the taxonomy. This includes a procedure to cluster local economies based on similarities between their industrial compositions, various analyses to interpret geographical and sectoral differences between the resulting clusters, and a comparison between their performance in various indicators produced to inform UK Government Levelling Up agendas. This section also contains an exploratory analysis of multi-sector companies including the sectors where they operate and their geography.

Section 6 concludes with a discussion of the implications of our analysis, issues for further research and development, and some observations about how the opportunities that we have identified in this programme of research could be leveraged by National Statistical Agencies and government

