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

## Colab Workflow 1: ruBERT Baseline

This workflow installs the package, clones the external data repositories,
generates `cointegrated/rubert-tiny2` embeddings, clusters the complete corpus,
prints scores as `metric - result - reference note`, saves score text files, and
writes an Excel workbook with cluster summaries under `/content/`.

```python
!git clone https://github.com/fortvivlan/cobaldclusterisation.git
%cd cobaldclusterisation
!pip install -e ".[embeddings,graph]" openpyxl
!pip install -U scikit-learn
```

Mount Google Drive before importing the baseline function if embeddings should
be saved there:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Clone or locate the corpus and hierarchy before importing `run`:

```python
from cobaldclusterisation import (
    corpus_to_dataframe,
    ensure_external_data,
    load_corpus,
)

paths = ensure_external_data()
print(paths.corpus_dir)
print(paths.hierarchy_csv)

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

Run the full ruBERT baseline. Algorithm names are case-insensitive. The
scalable defaults are `BisectingKMeans`, `BIRCH`, and `KNNLeiden`; legacy
options remain available by explicit request: `KMeans`, `MiniBatchKMeans`,
`Agglomerative`, and `HDBSCAN`. The pipeline does not sample before clustering.
By default, count-based algorithms use the number of SEMCLASS labels actually
present in the data (`data_semclass`; 565 for the bundled train+dev corpus).

```python
from cobaldclusterisation.rubert_baseline import run

rubert_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_embeddings_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    embedding_batch_size=32,
    clustering_batch_size=4096,
    device="cuda",
    seed=42,
    show_progress=True,
)

rubert_baseline["paths"]
```

`rubert_baseline["paths"]["excel"]` points to the `.xlsx` summary file under
`/content/`. Score tables are printed and also saved as `.txt` files listed in
`rubert_baseline["paths"]["scores_txt"]`. Metric-row score tables are saved as
CSV and Excel files at `rubert_baseline["paths"]["scores_csv"]` and
`rubert_baseline["paths"]["scores_xlsx"]`. Per-cluster hierarchy alignment
tables are saved at `rubert_baseline["paths"]["hierarchy_alignment_excel"]`.

To rerun clustering from a saved embedding pickle without loading the model
again, pass `embeddings_path`:

```python
rubert_baseline = run(
    embeddings_path="/content/drive/MyDrive/cobald_outputs/rubert_tiny2_cobald.pkl",
    hierarchy=paths.hierarchy_csv,
    algorithms=["BisectingKMeans", "BIRCH", "KNNLeiden"],
    n_clusters="data_semclass",
    output_dir="/content",
)
```

To run a legacy comparison, request the older algorithms explicitly:

```python
rubert_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    algorithms=["MiniBatchKMeans", "HDBSCAN"],
    n_clusters="data_semclass",
    min_cluster_size=25,
    min_samples=10,
    hdbscan_n_jobs=-1,
)
```

To test whether punctuation adds noise, run the same model with punctuation
removed from transformer contexts. The target token count stays the same.

```python
rubert_no_punct = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    algorithms=["BIRCH"],
    n_clusters="data_semclass",
    output_dir="/content",
    embedding_filename="rubert_tiny2_cobald_no_punct_context.pkl",
    excel_filename="rubert_tiny2_no_punct_cluster_summaries.xlsx",
    label="rubert_tiny2_no_punct",
    embedding_batch_size=32,
    device="cuda",
    include_punctuation_context=False,
    show_progress=True,
)
```

## Colab Workflow 2: SambaLingo Russian Base

`sambanovasystems/SambaLingo-Russian-Base` is a larger Llama-2-7B-style
pretrained Russian/English model. It can be used by the same embedding
pipeline, but it is much heavier than ruBERT tiny. On Colab, use a high-memory
GPU runtime, `float16`, and `batch_size=1`. The model repository is about 27.8
GB, and the resulting embeddings are also large.

Run the install, Drive, data, and helper cells from the ruBERT workflow first.
If Hugging Face requires authentication or license acceptance for the model,
log in before loading it:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Run the full SambaLingo baseline. The pipeline saves full embeddings, releases
CUDA memory, builds 256-dimensional IncrementalPCA clustering features, and
clusters the complete token set without sampling.

```python
from cobaldclusterisation.sambalingo_baseline import run

sambalingo_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_embeddings_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    projection_n_components=256,
    projection_batch_size=8192,
    embedding_batch_size=1,
    clustering_batch_size=4096,
    device="cuda",
    torch_dtype="float16",
    seed=42,
    show_progress=True,
)

sambalingo_baseline["paths"]
```

`sambalingo_baseline["paths"]["embeddings"]` points to the full SambaLingo
embedding pickle. `sambalingo_baseline["paths"]["clustering_features"]` points
to the reduced IncrementalPCA feature pickle used for clustering. The Excel
summary, metric-row score CSV/XLSX, and per-run score text files are saved
under `/content/`.

Pass `embeddings_path` to skip SambaLingo embedding generation and rerun only
projection, clustering, and exports from a saved full-embedding pickle.

Agglomerative clustering is disabled above 50,000 rows unless explicitly
overridden because it can require quadratic memory/time. To force it on the
full corpus:

```python
sambalingo_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    algorithms=["Agglomerative"],
    n_clusters=100,
    allow_quadratic_algorithms=True,
)
```

Cluster names are assigned by finding the actual token embedding closest to the
cluster centroid in the clustering feature space. Summaries include token
count, unique lemma count, all lemmas with counts, frequent forms, SEMCLASS
labels with counts, and example contexts. List fields use newline-separated
values so they are readable as wrapped cells in Excel.

To try another Hugging Face model in Colab, pass a different `model_name`.
For larger Russian LLM checkpoints, reduce `batch_size`, use a GPU runtime, and
set `trust_remote_code=True` only if the model repository requires it and you
trust that code.

```python
from cobaldclusterisation.embeddings import EmbeddingConfig

config = EmbeddingConfig(
    model_name="<huggingface-russian-base-model-id>",
    batch_size=1,
    device="cuda",
    torch_dtype="float16",
    trust_remote_code=True,
)
```

## Colab Workflow 3: GigaChat3 Base

`ai-sage/GigaChat3-10B-A1.8B-base` is a DeepSeek-V3-style causal language
model with 10B total and 1.8B active parameters. Its Hugging Face quickstart
uses `AutoModelForCausalLM` for generation, but the embedding baseline can use
the same hidden-state extraction pipeline as the other baselines because
Transformers exposes the underlying DeepSeek-V3 model hidden states. Use a
recent Transformers release, a high-memory GPU runtime, `bfloat16`, and
`batch_size=1`. The model repository is about 23 GB, and full embeddings are
saved before PCA projection.

Run the install, Drive, data, and helper cells from the ruBERT workflow first.
If Hugging Face requires authentication for the model, log in before loading
it:

```python
from huggingface_hub import notebook_login

notebook_login()
```

Run the full GigaChat3 baseline. The pipeline saves full embeddings, releases
CUDA memory, builds 256-dimensional IncrementalPCA clustering features, and
clusters the complete token set without sampling.

```python
from cobaldclusterisation.gigachat_baseline import run

gigachat_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_embeddings_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    projection_n_components=256,
    projection_batch_size=8192,
    embedding_batch_size=1,
    clustering_batch_size=4096,
    device="cuda",
    torch_dtype="bfloat16",
    trust_remote_code=False,
    seed=42,
    show_progress=True,
)

gigachat_baseline["paths"]
```

`gigachat_baseline["paths"]["embeddings"]` points to the full GigaChat3
embedding pickle. `gigachat_baseline["paths"]["clustering_features"]` points to
the reduced IncrementalPCA feature pickle used for clustering. The Excel
summary, metric-row score CSV/XLSX, and per-run score text files are saved
under `/content/`.
Agglomerative clustering has the same full-corpus guard as the SambaLingo
baseline.

Pass `embeddings_path` to skip GigaChat3 embedding generation and rerun only
projection, clustering, and exports from a saved full-embedding pickle.

## Colab Workflow 4: External FineWeb2 MiniBatch K-Means

This workflow is separate from the hierarchical-clustering code. It trains
`MiniBatchKMeans` on unlabeled Russian raw text from
`HuggingFaceFW/fineweb-2`, subset `rus_Cyrl`, then predicts cluster labels for
CoBaLD tokens and evaluates those predicted labels against `SEMCLASS`.
FineWeb2 is streamed, so the full Russian subset is not downloaded.

Install the additional streaming dependency:

```python
!pip install -e ".[embeddings,external]" openpyxl
```

The default external sample is 3,000,000 word-like target tokens. This is above
the CoBaLD train+dev embedding target count and is feasible in Colab because
features are written to disk-backed memmaps. For SambaLingo and GigaChat, the
pipeline fits IncrementalPCA before clustering so it does not need to store the
full 4096-dimensional external embedding matrix.

```python
from cobaldclusterisation.external_minibatch_baseline import run_rubert_external

external_rubert = run_rubert_external(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_models_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    max_train_tokens=3_000_000,
    n_clusters="data_semclass",
    embedding_batch_size=16,
    kmeans_batch_size=4096,
    device="cuda",
    seed=42,
    show_progress=True,
)

external_rubert["paths"]
```

For high-memory GPU runs:

```python
from cobaldclusterisation.external_minibatch_baseline import (
    run_gigachat_external,
    run_sambalingo_external,
)

external_sambalingo = run_sambalingo_external(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_models_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    max_train_tokens=3_000_000,
    device="cuda",
    show_progress=True,
)

external_gigachat = run_gigachat_external(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    output_dir="/content",
    save_models_to_drive=True,
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    max_train_tokens=3_000_000,
    device="cuda",
    show_progress=True,
)
```

Outputs include CoBaLD cluster summaries, hierarchy alignment, score CSV/XLSX,
a token-level CoBaLD table with the predicted `cluster` column, the fitted
K-Means model, optional PCA model, and a JSON run config.

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
  --algorithms bisecting_kmeans birch knn_leiden \
  --n-clusters data_semclass \
  --output outputs/clusters/scalable_defaults.pkl
```

The clustering CLI shows progress by default. Add `--no-progress` for quieter
batch runs. kNN graph clustering requires the optional `graph` extra.

Generated outputs under `outputs/` and pickle files are ignored by Git.
