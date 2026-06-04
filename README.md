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
!pip install -e ".[embeddings]" openpyxl
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

Run the full ruBERT baseline. Algorithm names are case-insensitive; supported
values are `KMeans`, `MiniBatchKMeans`, `Agglomerative`, and `HDBSCAN`. The
pipeline does not sample before clustering.

```python
from cobaldclusterisation.rubert_baseline import run

rubert_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    algorithms=["KMeans", "MiniBatchKMeans"],
    n_clusters=100,
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
`rubert_baseline["paths"]["scores_txt"]`. A compact score CSV is saved as
`rubert_baseline["paths"]["scores_csv"]`.

To include HDBSCAN in the complete-corpus run:

```python
rubert_baseline = run(
    data_dir=paths.corpus_dir,
    hierarchy=paths.hierarchy_csv,
    algorithms=["KMeans", "MiniBatchKMeans", "HDBSCAN"],
    n_clusters=100,
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
    algorithms=["MiniBatchKMeans"],
    n_clusters=100,
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

Generate SambaLingo embeddings with conservative settings:

```python
from cobaldclusterisation.embeddings import EmbeddingConfig, generate_embeddings_from_corpus

sambalingo_config = EmbeddingConfig(
    model_name="sambanovasystems/SambaLingo-Russian-Base",
    batch_size=1,
    device="cuda",
    torch_dtype="float16",
    max_length=512,
    include_punctuation_context=True,
)

sambalingo_payload = generate_embeddings_from_corpus(
    data_dir=paths.corpus_dir,
    output_path="sambalingo_russian_base_cobald.pkl",
    drive_dir="/content/drive/MyDrive/cobald_outputs",
    config=sambalingo_config,
    show_progress=True,
)

sambalingo_payload["embeddings"].shape
```

Release CUDA memory after embedding extraction before clustering:

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

Use reproducible samples for resource-heavy clustering comparisons. The
K-Means sample can be larger; the all-algorithm sample is smaller because
Agglomerative clustering and HDBSCAN can be much more memory-intensive.

```python
def sample_embedding_payload(payload, sample_size=20000, seed=42):
    total = len(payload["tokens"])
    if sample_size is None or sample_size >= total:
        return payload
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=sample_size, replace=False))
    return {
        **payload,
        "embeddings": payload["embeddings"][indices],
        "tokens": payload["tokens"].iloc[indices].reset_index(drop=True),
    }


sambalingo_kmeans_sample = sample_embedding_payload(
    sambalingo_payload,
    sample_size=20000,
    seed=42,
)
sambalingo_algorithm_sample = sample_embedding_payload(
    sambalingo_payload,
    sample_size=5000,
    seed=42,
)
len(sambalingo_kmeans_sample["tokens"]), len(sambalingo_algorithm_sample["tokens"])
```

Run K-Means on the sample and download the cluster summary workbook:

```python
from cobaldclusterisation.clustering import ClusterConfig, run_clustering

sambalingo_kmeans_result = run_clustering(
    sambalingo_kmeans_sample["embeddings"],
    sambalingo_kmeans_sample["tokens"],
    config=ClusterConfig(
        algorithm="kmeans",
        n_clusters=100,
        random_state=42,
        n_init=5,
    ),
    hierarchy=paths.hierarchy_csv,
    hierarchy_depths=(1, 2, 3),
    show_progress=True,
)

sambalingo_kmeans_excel = write_cluster_summary_excel(
    [sambalingo_kmeans_result],
    "outputs/clusters/sambalingo_kmeans_cluster_summary.xlsx",
)
files.download(sambalingo_kmeans_excel)

print_result_scores("SambaLingo K-Means sample", sambalingo_kmeans_result)
sambalingo_kmeans_result["summary"].head(20)
```

Run all available clustering algorithms on the smaller sample. Lower
`sample_size` above if the runtime runs out of memory.

```python
from cobaldclusterisation.clustering import run_clustering_suite

sambalingo_all_configs = [
    ClusterConfig(
        algorithm="minibatch_kmeans",
        n_clusters=100,
        random_state=42,
        batch_size=4096,
    ),
    ClusterConfig(
        algorithm="kmeans",
        n_clusters=100,
        random_state=42,
        n_init=5,
    ),
    ClusterConfig(
        algorithm="agglomerative",
        n_clusters=100,
    ),
    ClusterConfig(
        algorithm="hdbscan",
        min_cluster_size=25,
        min_samples=10,
        metric="euclidean",
        n_jobs=-1,
    ),
]

sambalingo_all_results = run_clustering_suite(
    sambalingo_algorithm_sample,
    configs=sambalingo_all_configs,
    hierarchy=paths.hierarchy_csv,
    hierarchy_depths=(1, 2, 3),
    show_progress=True,
)

for index, result in enumerate(sambalingo_all_results, start=1):
    print_result_scores(f"SambaLingo all algorithms run {index}", result)
    print()
```

Cluster names are assigned by finding the actual token embedding closest to the
cluster centroid. Summaries include token count, unique lemma count, frequent
lemmas, frequent forms, all SEMCLASS labels found in the cluster, and example
contexts. List fields use newline-separated values so they are readable as
wrapped cells in Excel.

```python
sambalingo_all_excel = write_cluster_summary_excel(
    sambalingo_all_results,
    "outputs/clusters/sambalingo_all_algorithms_cluster_summaries.xlsx",
)
files.download(sambalingo_all_excel)
```

`run_clustering_suite(..., show_progress=True)` displays a `tqdm` progress bar
in Colab while algorithm configurations are running. `run_clustering` also has
a smaller progress bar for the fitting, scoring, and summary stages of one run.

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
