# cobaldclusterisation

This repository contains importable Python helpers for exploring whether
embedding-based clustering can recover or approximate manually assigned CoBaLD
semantic classes.

The corpus and semantic hierarchy live in external GitHub repositories and are
not committed here:

```text
CobaldRus/
  train.conllu
  dev.conllu
semantic-hierarchy/
  hyperonims_hierarchy.csv
```

`train.conllu` and `dev.conllu` are merged at load time. Each token row must
have 12 columns: `ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`,
`DEPREL`, `DEPS`, `MISC`, `DEEPSLOT`, and `SEMCLASS`.

## Install In Colab

```python
!git clone https://github.com/fortvivlan/cobaldclusterisation.git
%cd cobaldclusterisation
!pip install -e ".[embeddings]"
```

Clone the external data repositories into the project directory:

```python
!git clone https://github.com/CobaldAnnotation/CobaldRus.git
!git clone https://github.com/CobaldAnnotation/semantic-hierarchy.git
```

Alternatively, let the package clone them if they are missing:

```python
from cobaldclusterisation import ensure_external_data

paths = ensure_external_data()
print(paths.corpus_dir)
print(paths.hierarchy_csv)
```

If embeddings should be saved to Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

## Parse The Corpus

```python
from cobaldclusterisation import corpus_to_dataframe, ensure_external_data, load_corpus

paths = ensure_external_data()

sentences = load_corpus(paths.corpus_dir, splits=("train", "dev"))
all_rows = corpus_to_dataframe(sentences)
embedding_targets = corpus_to_dataframe(
    sentences,
    include_punctuation=False,
    include_empty=False,
)

print(len(sentences))
print(len(all_rows))
print(len(embedding_targets))
embedding_targets.head()
```

The default embedding target policy is surface non-punctuation tokens only.
Decimal-ID ellipsis rows such as `2.1 #NULL` are parsed but are not embedding
targets.

## Generate ruBERT Tiny Embeddings

The baseline model is `cointegrated/rubert-tiny2`. For tokens split into
subwords, the token embedding is the mean of its subword hidden states.

```python
from cobaldclusterisation.embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
)

config = EmbeddingConfig(
    model_name="cointegrated/rubert-tiny2",
    batch_size=32,
    device="cuda",
    max_length=512,
    include_punctuation_context=True,
)

payload = generate_embeddings_from_corpus(
    data_dir=paths.corpus_dir,
    output_path="rubert_tiny2_cobald.pkl",
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    config=config,
)

payload["embeddings"].shape, payload["tokens"].head()
```

To test whether punctuation adds noise, run the same model with punctuation
removed from transformer contexts. The target token count stays the same.

```python
no_punct_config = EmbeddingConfig(
    model_name="cointegrated/rubert-tiny2",
    batch_size=32,
    device="cuda",
    max_length=512,
    include_punctuation_context=False,
)

no_punct_payload = generate_embeddings_from_corpus(
    data_dir=paths.corpus_dir,
    output_path="rubert_tiny2_cobald_no_punct_context.pkl",
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    config=no_punct_config,
)
```

To try another Hugging Face model in Colab, pass a different `model_name`.
For larger Russian LLM checkpoints, reduce `batch_size`, use a GPU runtime, and
set `trust_remote_code=True` only if the model repository requires it and you
trust that code.

```python
config = EmbeddingConfig(
    model_name="<huggingface-russian-base-model-id>",
    batch_size=1,
    device="cuda",
    torch_dtype="float16",
    trust_remote_code=True,
)
```

## Cluster And Evaluate

```python
from cobaldclusterisation.clustering import (
    ClusterConfig,
    run_clustering,
    scores_to_dataframe,
)
from cobaldclusterisation.embeddings import load_embeddings_pickle

payload = load_embeddings_pickle("/content/drive/MyDrive/cobald_outputs/rubert_tiny2_cobald.pkl")

result = run_clustering(
    payload["embeddings"],
    payload["tokens"],
    config=ClusterConfig(
        algorithm="minibatch_kmeans",
        n_clusters=100,
        random_state=42,
    ),
    hierarchy=paths.hierarchy_csv,
    hierarchy_depths=(1, 2, 3),
    show_progress=True,
)

result["scores"]
result["summary"].head(20)
```

Cluster names are assigned by finding the actual token embedding closest to the
cluster centroid. Summaries include cluster size, representative token, frequent
lemmas, frequent semantic classes, and example contexts.

For several scalable clustering runs:

```python
from cobaldclusterisation.clustering import default_cluster_configs, run_clustering_suite

results = run_clustering_suite(
    payload,
    configs=default_cluster_configs(n_clusters=(50, 100)),
    hierarchy=paths.hierarchy_csv,
    show_progress=True,
)

scores_to_dataframe(results)
```

`run_clustering_suite(..., show_progress=True)` displays a `tqdm` progress bar
in Colab while algorithm configurations are running. `run_clustering` also has
a smaller progress bar for the fitting, scoring, and summary stages of one run.

`AgglomerativeClustering` and `DBSCAN` are available through `ClusterConfig`,
but they can be expensive on the full corpus. Use them first on a smaller
sample.

## CLI Examples

After `pip install -e ".[embeddings]"`:

```bash
cobald parse --data-dir CobaldRus

cobald embed \
  --data-dir CobaldRus \
  --model-name cointegrated/rubert-tiny2 \
  --batch-size 32 \
  --device cuda \
  --output outputs/embeddings/rubert_tiny2.pkl

cobald embed \
  --data-dir CobaldRus \
  --model-name cointegrated/rubert-tiny2 \
  --batch-size 32 \
  --device cuda \
  --no-punctuation-context \
  --output outputs/embeddings/rubert_tiny2_no_punct_context.pkl

cobald cluster outputs/embeddings/rubert_tiny2.pkl \
  --hierarchy semantic-hierarchy \
  --algorithms minibatch_kmeans \
  --n-clusters 100 \
  --output outputs/clusters/minibatch_k100.pkl
```

The clustering CLI shows progress by default. Add `--no-progress` for quieter
batch runs.

Generated outputs under `outputs/` and pickle files are ignored by Git.
