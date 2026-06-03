# Project Notes

This project explores a semi-automatic semantic hierarchy for the CoBaLD
annotation standard. The working hypothesis is that embedding-based clustering
can provide evidence for hierarchy design that is normally produced manually by
human linguists.

## Working Style

- Prefer small, importable Python modules over notebook-only code.
- Scripts should be suitable for importing and running from Google Colaboratory.
- Keep computational steps reproducible: expose parameters explicitly, avoid
  hidden global state, and make random seeds configurable where relevant.
- Favor clear data-frame transformations, typed helper functions, and explicit
  intermediate outputs for linguistic inspection.
- Treat clustering output as evidence for linguists, not as an unquestioned
  final hierarchy.

## Expected Components

Likely modules may include:

- loading CoBaLD annotation data;
- generating or loading embeddings;
- clustering embeddings;
- evaluating and visualizing cluster structure;
- exporting candidate hierarchy tables for human review.

## Repository Conventions

- Always use Unicode.
- Keep Colab-facing scripts dependency-light and document any required packages.
- Do not commit generated caches, notebook checkpoints, model artifacts, or large
  intermediate data unless explicitly requested.
