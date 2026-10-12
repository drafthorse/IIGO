# IIGO algorithms and manuscript results

Algorithm implementations, experiment scripts and numerical results accompanying the IIGO manuscript. Files are stored directly in the folders below.

## Files

- `01_code`: the final IIGO, the 15 competing algorithms, ablation and sensitivity variants, engineering models and experiment scripts.
- `input`: only the CEC input files required for the manuscript's 30 dimensional and 100 dimensional experiments.
- `02_data`: original numerical run records, seeds, statistics and figure source data used in the manuscript.
- `03_protocols/paper_settings.json`: experiment settings used in the paper.

## Result locations

| Manuscript | Folder |
| --- | --- |
| Table 2 reported validation summary | `02_data/00_diversity_validation` |
| Tables 4 to 6 and Supplementary Tables S3 and S4 | `02_data/01_main_benchmark_NP10` |
| Table 7 | `02_data/03_high_dimensional_D100` |
| Table 8 and Supplementary Tables S1 and S2 | `02_data/04_ablation` |
| Tables 9 and 10 | `02_data/02_population_and_configuration` |
| Table 11 | `02_data/05_parameter_sensitivity` |
| Table 12 | `02_data/06_runtime` |
| Figures 5 to 9 | `02_data/07_figure_source_data` |
| Tables 13 to 16 and Table 20 | `02_data/08_engineering/classical_and_ELD_300000FEs` |
| Table 18 | `02_data/08_engineering/truss_recommended_FEs` |

Numerical records retain their stored precision. Experimental seeds are included with each experiment.

The Table 2 CSV stores the reported validation summary, rather than individual angular samples.
