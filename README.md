# GHI Forecast Benchmark — 4 Sites (Bengkulu, Banten, Kalimantan Barat, Jambi)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21833225.svg)](https://doi.org/10.5281/zenodo.21833225)

Harmonised benchmark scripts and per-site result summaries for the GHI
(Global Horizontal Irradiance) forecasting study across four Indonesian
sites. Prepared as supplementary material for the associated dissertation
paper submission.

## Structure

```
bengkulu/
  duckdb_bengkulu/       audit & anchor-count scripts, SYNOP null audit
  ghi_forecast_pipeline/ numbered end-to-end pipeline scripts (01-.. build/train/walkforward)
banten/                  audit scripts, R1/R8 outputs, learning curve, meteo isolation
kalbar/                  build/train scripts, R1/R8 outputs, anchor audits
jambi/                   build/train/ablation scripts, R1/R8 outputs
```

Each site folder contains its data-pipeline and model scripts plus the
`outputs_*` result summaries (metrics, audit reports, markdown summaries)
produced by those scripts.

## Data availability

Raw datasets (DuckDB databases, parquet feature tables, trained model
binaries) are not included in this repository due to size (multi-GB per
site). They are available from the corresponding author on request.
Row-level test-set prediction dumps referenced by some scripts are
likewise omitted for size reasons; aggregate metrics are included under
each site's `outputs_*` folders.

## Reproducing results

Scripts expect a local or MotherDuck-hosted DuckDB database per site
(`MOTHERDUCK_TOKEN` environment variable, or a local `.duckdb` file — see
individual script headers for connection details).

## Citation

Archived releases of this repository are permanently citable via Zenodo:
[10.5281/zenodo.21833225](https://doi.org/10.5281/zenodo.21833225).
