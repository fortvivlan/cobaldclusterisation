# CoBaLD Clusterisation Colab API Reference

This document lists the public functions and dataclass constructors that are
intended to be imported from Google Colaboratory notebooks. Private helpers
whose names start with `_` are intentionally omitted.

Most functions return plain Python dictionaries, `pandas.DataFrame` objects,
NumPy arrays, or filesystem paths, so intermediate results can be inspected in
Colab before deciding whether a clustering result is linguistically useful.

## Recommended Colab Setup

```python
!git clone https://github.com/fortvivlan/cobaldclusterisation.git
%cd cobaldclusterisation
!pip install -e ".[embeddings,graph,external]" openpyxl
!pip install -U scikit-learn

from google.colab import drive
drive.mount("/content/drive")
```

Default Drive output root:

```python
DRIVE_DIR = "/content/drive/MyDrive/cobald_outputs"
```

The current baseline functions use this output convention:

- Embedding and clustering feature pickle files are saved in `DRIVE_DIR` when
  Drive saving is enabled.
- Result files are saved in `DRIVE_DIR/results` when `output_dir="/content"`
  and `results_dir` is not passed.
- Local or test runs with a custom `output_dir` write result files into that
  custom directory unless `results_dir` is passed explicitly.

Result files include:

- `{prefix}_summary.xlsx`: short cluster summaries. `semclass_lemma_examples`
  contains up to 50 lemma examples per semantic class.
- `{prefix}_semclass_cluster_map.xlsx`: semantic-class rows with the automatic
  clusters containing each class and the number of such clusters.
- `{prefix}_scores.xlsx` and `{prefix}_scores.csv`: metric-row score tables.
- `{prefix}_*_scores.txt`: printed score tables for individual runs.
- `{prefix}_hierarchy_alignment.xlsx`: per-cluster alignment to hierarchy
  depths when a hierarchy is available.
- `{prefix}_artifacts.pkl`: reusable pickle with labels, annotated CoBaLD token
  table, summaries, scores, hierarchy alignment, configs, and metadata.
- `{prefix}_{run}_annotated.conllu`: CoNLL-U Plus target-token export with an
  added `AUTO_SEMCLASS` column.

Prefixes are built from requested cluster counts, model label, and clustering
family, for example:

- `100cl_rubert_tiny2_Minibatch_Kmeans`
- `100,200,300cl_rubert_tiny2_Hierarchical`

For annotator-facing explanations of every workbook column and the CoNLL-U Plus
exports, see [output_columns_guide.md](output_columns_guide.md).

## Resource Helpers

Import:

```python
from cobaldclusterisation import ensure_external_data, resolve_hierarchy_path
from cobaldclusterisation.resources import ExternalDataPaths
```

### `ExternalDataPaths(corpus_dir, hierarchy_repo_dir, hierarchy_csv)`

Dataclass returned by `ensure_external_data`.

Arguments:

- `corpus_dir`: path to the local CoBaLD corpus clone containing
  `train.conllu` and `dev.conllu`.
- `hierarchy_repo_dir`: path to the semantic hierarchy repository clone.
- `hierarchy_csv`: path to `hyperonims_hierarchy.csv`.

### `resolve_hierarchy_path(path)`

Find a hierarchy CSV from either a direct file path or a directory path.

Arguments:

- `path`: file path to a hierarchy CSV, or directory containing
  `hyperonims_hierarchy.csv`.

Returns: `Path`.

### `ensure_external_data(...)`

Clone or reuse the external CoBaLD and hierarchy repositories.

Signature:

```python
ensure_external_data(
    *,
    root_dir=".",
    corpus_dir_name="CobaldRus",
    hierarchy_dir_name="semantic-hierarchy",
    corpus_repo="https://github.com/CobaldAnnotation/CobaldRus.git",
    hierarchy_repo="https://github.com/CobaldAnnotation/semantic-hierarchy.git",
)
```

Arguments:

- `root_dir`: directory where external repositories should be located.
- `corpus_dir_name`: local folder name for the CoBaLD corpus repository.
- `hierarchy_dir_name`: local folder name for the hierarchy repository.
- `corpus_repo`: Git URL for the CoBaLD corpus.
- `hierarchy_repo`: Git URL for the semantic hierarchy.

Returns: `ExternalDataPaths`.

## CoNLL-U Data Helpers

Import:

```python
from cobaldclusterisation import (
    CONLLU_COLUMNS,
    Sentence,
    Token,
    corpus_to_dataframe,
    iter_tokens,
    load_corpus,
    load_semclass_hierarchy,
    parse_conllu,
    tokens_to_dataframe,
)
from cobaldclusterisation.data import parse_ud_conllu, load_ud_corpus
```

### `Token(...)`

Dataclass representing one CoBaLD token row.

Constructor arguments:

- `id`, `form`, `lemma`, `upos`, `xpos`, `feats`, `head`, `deprel`, `deps`,
  `misc`, `deepslot`, `semclass`: the 12 source columns.
- `sentence_id`: sentence id from CoNLL-U comments, if known.
- `split`: corpus split such as `train`, `dev`, or `syntagrus`.
- `sentence_index`: numeric sentence position within the loaded split.
- `token_index`: numeric token-row position within the sentence.

Useful properties:

- `is_punctuation`: true when `UPOS` is `PUNCT`.
- `is_empty_node`: true for empty/ellipsis rows such as decimal IDs.
- `is_multiword_token`: true for CoNLL-U multiword-token range rows.
- `is_surface_token`: true for explicit non-empty, non-multiword token rows.
- `has_semclass`: true when `SEMCLASS` is not empty and not `_`.

### `Sentence(tokens, metadata={}, comments=[], split=None, index=None)`

Dataclass representing a parsed sentence.

Arguments:

- `tokens`: ordered list of `Token` rows.
- `metadata`: parsed comment metadata such as `sent_id` and `text`.
- `comments`: original comment lines.
- `split`: corpus split name.
- `index`: sentence index in the loaded split.

Useful methods/properties:

- `sent_id`: metadata `sent_id`, if present.
- `text`: metadata `text`, or a simple token reconstruction.
- `surface_tokens(include_punctuation=True)`: explicit surface rows.
- `context_text(include_punctuation=True)`: text used for embedding context.
- `embedding_targets()`: surface non-punctuation tokens.
- `to_dataframe()`: one-sentence DataFrame.

### `parse_conllu(path, *, split=None)`

Parse a 12-column CoBaLD CoNLL-U-like file.

Arguments:

- `path`: file path to a CoBaLD `.conllu` file.
- `split`: optional split label to attach to tokens and sentences.

Returns: `list[Sentence]`.

### `load_corpus(data_dir="CobaldRus", *, splits=("train", "dev"))`

Load multiple CoBaLD split files from a corpus directory.

Arguments:

- `data_dir`: directory containing files like `train.conllu`.
- `splits`: split names to load; each maps to `{split}.conllu`.

Returns: `list[Sentence]`.

### `parse_ud_conllu(path, *, split="syntagrus")`

Parse a standard 10-column UD CoNLL-U file into project `Sentence` objects.

Arguments:

- `path`: file path to a UD `.conllu` file.
- `split`: split label to attach to tokens.

Returns: `list[Sentence]`. `DEEPSLOT` and `SEMCLASS` are filled with `_`.

### `load_ud_corpus(paths, *, split="syntagrus")`

Load and merge several UD CoNLL-U files.

Arguments:

- `paths`: sequence of UD `.conllu` file paths.
- `split`: split label to attach to tokens.

Returns: `list[Sentence]`.

### `iter_tokens(sentences, *, include_punctuation=False, include_empty=False)`

Iterate tokens from loaded sentences.

Arguments:

- `sentences`: iterable of `Sentence` objects.
- `include_punctuation`: include `UPOS=PUNCT` rows when true.
- `include_empty`: include empty/ellipsis rows when true.

Returns: iterator of `Token`.

### `tokens_to_dataframe(tokens, *, context_text=None)`

Convert tokens to a `pandas.DataFrame`.

Arguments:

- `tokens`: iterable of `Token` objects.
- `context_text`: optional text value to attach to every row.

Returns: DataFrame with original CoBaLD columns plus metadata columns.

### `corpus_to_dataframe(sentences, *, include_punctuation=True, include_empty=True)`

Convert a corpus to a DataFrame.

Arguments:

- `sentences`: iterable of `Sentence` objects.
- `include_punctuation`: include punctuation rows when true.
- `include_empty`: include empty/ellipsis rows when true.

Returns: DataFrame. For embedding targets, use
`include_punctuation=False, include_empty=False`.

### `load_semclass_hierarchy(path)`

Load the semantic hierarchy CSV.

Arguments:

- `path`: path to `hyperonims_hierarchy.csv`.

Returns: hierarchy DataFrame.

## Embedding Helpers

Import:

```python
from cobaldclusterisation.embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    generate_token_embeddings,
    load_embeddings_pickle,
    save_embeddings_pickle,
)
```

### `EmbeddingConfig(...)`

Embedding-generation configuration.

Arguments:

- `model_name`: Hugging Face model id.
- `batch_size`: number of sentences per transformer batch.
- `max_length`: tokenizer truncation length.
- `layer`: hidden-state layer to use; `-1` uses `last_hidden_state`.
- `device`: `cuda`, `cpu`, or `None` for automatic selection.
- `seed`: random seed for reproducibility.
- `normalize`: L2-normalize individual token embeddings when true.
- `trust_remote_code`: passed to Hugging Face model/tokenizer loading.
- `torch_dtype`: optional dtype name such as `float16` or `bfloat16`.
- `include_punctuation_context`: keep punctuation tokens in transformer
  context when true. Target tokens remain surface non-punctuation tokens.

### `generate_token_embeddings(sentences, *, config=None, show_progress=True)`

Generate contextual embeddings for already loaded sentences.

Arguments:

- `sentences`: sequence of `Sentence` objects.
- `config`: `EmbeddingConfig`; defaults to ruBERT tiny2 settings.
- `show_progress`: show tqdm progress bars.

Returns: payload dictionary with:

- `embeddings`: NumPy float32 matrix.
- `tokens`: target-token DataFrame aligned row-for-row with embeddings.
- `config`: embedding configuration metadata.

### `save_embeddings_pickle(payload, output_path, *, drive_dir=None)`

Save an embedding payload.

Arguments:

- `payload`: dictionary with at least embeddings and tokens.
- `output_path`: desired output file path.
- `drive_dir`: optional Drive directory. When passed, the file is saved as
  `Path(drive_dir) / Path(output_path).name`.

Returns: saved `Path`.

### `load_embeddings_pickle(path)`

Load a saved embedding payload.

Arguments:

- `path`: pickle path.

Returns: payload dictionary.

### `generate_embeddings_from_corpus(...)`

Load CoBaLD splits, generate embeddings, save them, and return the payload.

Signature:

```python
generate_embeddings_from_corpus(
    *,
    data_dir="CobaldRus",
    output_path="outputs/embeddings/rubert_tiny2.pkl",
    drive_dir=None,
    splits=("train", "dev"),
    config=None,
    show_progress=True,
)
```

Arguments:

- `data_dir`: CoBaLD corpus directory.
- `output_path`: local output path or filename.
- `drive_dir`: optional Drive directory for the saved pickle.
- `splits`: split files to load.
- `config`: `EmbeddingConfig`.
- `show_progress`: show tqdm progress bars.

Returns: embedding payload with `saved_path`.

## Collocation Helpers

Import:

```python
from cobaldclusterisation.collocations import (
    CollocationClusterConfig,
    CollocationConfig,
    build_collocation_tables,
    run_collocation_cluster_experiment,
    run_collocation_experiment,
)
```

Install:

```python
!pip install -e ".[collocations]" openpyxl
```

These helpers create sentence-bounded bigram and trigram tables from CoBaLD
surface non-punctuation tokens. By default they produce both lemma-based and
surface-form sheets, lowercase token text, require `min_freq=3`, and sort by
PMI while keeping frequency, raw frequency, likelihood ratio, Student's t, and
chi-square columns.

### `CollocationConfig(...)`

Configuration for one collocation export.

Arguments:

- `data_dir`: CoBaLD corpus directory.
- `splits`: split files to load, default `("train", "dev")`.
- `token_bases`: `("lemma", "form")` by default.
- `ngram_sizes`: `(2, 3)` by default.
- `min_freq`: minimum n-gram count before scoring, default `3`.
- `sort_by`: ranking column, default `pmi`.
- `max_rows`: optional row cap per sheet.
- `lowercase`: lowercase token text before counting when true.
- `output_path`: output `.xlsx` workbook path.

### `build_collocation_tables(sentences, ...)`

Build collocation DataFrames from already loaded `Sentence` objects.

Returns: dictionary keyed by sheet-style names such as `lemma_bigrams` and
`form_trigrams`.

### `run_collocation_experiment(config=None, **kwargs)`

Load CoBaLD data, build collocation tables, write the Excel workbook, and
return the intermediate objects.

Example:

```python
from cobaldclusterisation.collocations import run_collocation_experiment

result = run_collocation_experiment(
    data_dir="CobaldRus",
    output_path="outputs/collocations/cobald_collocations.xlsx",
)

result["tables"]["lemma_bigrams"].head()
```

Returns: dictionary with `config`, `sentences`, `tables`, `metadata`, and
`output_path`.

### `CollocationClusterConfig(...)`

Configuration for comparing collocations to saved clustering artifacts.

Arguments are the same collocation settings as `CollocationConfig`, plus:

- `artifact_path`: path to a clustering `*_artifacts.pkl` file.
- `output_path`: optional cluster-overlap workbook path. When omitted, the
  workbook is written next to the artifact in the Drive `results` folder with
  the artifact prefix.
- `max_examples_per_run`: capped exact collocation examples per clustering run.
- `create_plots`: write optional diagnostic PNG plots when `matplotlib` is
  installed.

### `run_collocation_cluster_experiment(config=None, **kwargs)`

Load CoBaLD data, compute collocations, load the clustering artifact, and write
an Excel workbook showing whether collocation parts share automatic clusters.
The workbook contains an overview sheet plus per-run sheets for collocations,
collocation-part pairs, and exact sentence examples. Rows include per-part
`SEMCLASS` summaries and both exact occurrence-level and aggregate lemma/form
same-cluster rates.

Example:

```python
from cobaldclusterisation.collocations import run_collocation_cluster_experiment

result = run_collocation_cluster_experiment(
    data_dir="CobaldRus",
    artifact_path=(
        "/content/drive/MyDrive/cobald_outputs/results/"
        "100,200,300,400,512,565cl_rubert_tiny2_Minibatch_Kmeans_artifacts.pkl"
    ),
    min_freq=3,
)

result["output_path"]
result["plot_paths"]
```

## Identical-Form SEMCLASS Similarity Helpers

Import:

```python
from cobaldclusterisation.form_semclass_similarity import (
    FormSemclassSimilarityConfig,
    run_form_semclass_similarity_experiment,
)
```

These helpers analyze case-insensitive identical `FORM` values that carry more
than one valid `SEMCLASS`. They combine an embedding payload pickle with a
clustering artifact pickle, compute exact aggregate cosine similarities, and
write one Excel workbook with separate sheets for every clustering run/count.

### `FormSemclassSimilarityConfig(...)`

Configuration for one identical-form analysis.

Arguments:

- `artifact_path`: clustering `*_artifacts.pkl` path.
- `embeddings_path`: optional aligned embedding payload pickle. If omitted,
  metadata and the Drive root are checked for `rubert_tiny2_cobald.pkl`.
- `output_path`: optional output workbook path.
- `max_examples_per_run`: capped context-example rows per clustering run.
- `examples_per_form`: maximum examples per selected ambiguous form.
- `create_plots`: save optional matplotlib scatter plots when available.

### `run_form_semclass_similarity_experiment(config=None, **kwargs)`

Load the artifact and embedding payload, validate row alignment, compute
identical-form cosine and cluster-distribution summaries, and write the result
workbook.

Example:

```python
result = run_form_semclass_similarity_experiment(
    artifact_path="drive/MyDrive/cobald_outputs/results/100,200,300,400,512,565cl_rubert_tiny2_Minibatch_Kmeans_artifacts.pkl",
    embeddings_path="drive/MyDrive/cobald_outputs/rubert_tiny2_cobald.pkl",
)
```

Returns: dictionary with `artifact_path`, `embeddings_path`, `run_tables`,
`output_path`, and `plot_paths`.

## Clustering Helpers

Import:

```python
from cobaldclusterisation.clustering import (
    ClusterConfig,
    default_cluster_configs,
    evaluate_clusters,
    fit_predict_clusters,
    fit_training_predict_evaluation,
    hierarchy_alignment_table,
    hierarchy_ancestor_labels,
    infer_semclass_cluster_count,
    purity_score,
    run_clustering,
    run_clustering_suite,
    run_clustering_with_training_data,
    save_clustering_results,
    scores_to_dataframe,
    summarize_clusters,
)
```

### `ClusterConfig(...)`

Configuration for one clustering run.

Arguments:

- `algorithm`: one of `kmeans`, `minibatch_kmeans`, `bisecting_kmeans`,
  `birch`, `knn_leiden`, `knn_louvain`, `agglomerative`, or `hdbscan`.
- `n_clusters`: requested cluster count for count-based algorithms.
- `random_state`: random seed.
- `normalize`: L2-normalize embeddings before clustering/scoring.
- `batch_size`: MiniBatchKMeans batch size.
- `n_init`: KMeans/MiniBatchKMeans/BisectingKMeans initialization count.
- `min_samples`: HDBSCAN minimum samples.
- `min_cluster_size`: HDBSCAN minimum cluster size.
- `cluster_selection_epsilon`: HDBSCAN selection epsilon.
- `cluster_selection_method`: HDBSCAN selection method.
- `allow_single_cluster`: HDBSCAN option.
- `n_jobs`: optional worker count for supported algorithms.
- `linkage`: agglomerative linkage.
- `metric`: distance metric for relevant algorithms.
- `birch_threshold`: BIRCH threshold.
- `birch_branching_factor`: BIRCH branching factor.
- `bisecting_strategy`: BisectingKMeans split strategy.
- `graph_n_neighbors`: neighbor count for kNN graph clustering.
- `graph_resolution`: Leiden/Louvain graph resolution.

### `fit_predict_clusters(embeddings, *, config=None)`

Fit one clustering model and return labels.

Arguments:

- `embeddings`: 2D NumPy array.
- `config`: `ClusterConfig`.

Returns: `(labels, model)`.

### `fit_training_predict_evaluation(training_embeddings, evaluation_embeddings, *, config=None)`

Fit on training embeddings and predict labels for evaluation embeddings.

Arguments:

- `training_embeddings`: matrix used to fit the model.
- `evaluation_embeddings`: matrix to label after fitting.
- `config`: `ClusterConfig`.

Returns: `(labels, model, prediction_method)`.

### `default_cluster_configs(*, n_clusters=(565,), random_state=42)`

Build default scalable clustering configs.

Arguments:

- `n_clusters`: sequence of counts.
- `random_state`: random seed.

Returns: list of `ClusterConfig`.

### `purity_score(y_true, y_pred)`

Compute cluster purity against gold labels.

Arguments:

- `y_true`: gold labels.
- `y_pred`: predicted labels.

Returns: float.

### `hierarchy_ancestor_labels(semclasses, hierarchy, *, depth)`

Map SEMCLASS labels to ancestor labels at a hierarchy depth.

Arguments:

- `semclasses`: sequence of SEMCLASS names.
- `hierarchy`: hierarchy DataFrame.
- `depth`: desired ancestor depth.

Returns: NumPy array of labels.

### `infer_semclass_cluster_count(tokens)`

Count usable SEMCLASS labels in a token DataFrame.

Arguments:

- `tokens`: DataFrame with a `SEMCLASS` column.

Returns: integer cluster count.

### `hierarchy_alignment_table(labels, tokens, hierarchy, *, hierarchy_depths=(1, 2, 3, 4, 5, 6, 7))`

Summarize how automatic clusters align with hierarchy labels.

Arguments:

- `labels`: predicted cluster labels aligned with `tokens`.
- `tokens`: token DataFrame with `SEMCLASS`.
- `hierarchy`: hierarchy DataFrame or path.
- `hierarchy_depths`: depths to evaluate.

Returns: DataFrame.

### `evaluate_clusters(...)`

Compute internal and SEMCLASS-aware clustering scores.

Signature:

```python
evaluate_clusters(
    embeddings,
    labels,
    *,
    tokens=None,
    hierarchy=None,
    hierarchy_depths=(1, 2, 3),
    max_silhouette_samples=10000,
    random_state=42,
)
```

Arguments:

- `embeddings`: 2D matrix used for scoring.
- `labels`: cluster labels.
- `tokens`: optional token DataFrame for SEMCLASS-aware scores.
- `hierarchy`: optional hierarchy DataFrame or path.
- `hierarchy_depths`: depths for hierarchy-aware external scores.
- `max_silhouette_samples`: sample cap for silhouette score.
- `random_state`: seed for score sampling.

Returns: score dictionary.

### `summarize_clusters(...)`

Build human-readable cluster summary rows.

Signature:

```python
summarize_clusters(
    embeddings,
    labels,
    tokens,
    *,
    examples_per_cluster=5,
    top_n=8,
    semclass_lemma_examples=50,
    random_state=42,
)
```

Arguments:

- `embeddings`: 2D matrix.
- `labels`: cluster labels.
- `tokens`: token DataFrame aligned with embeddings.
- `examples_per_cluster`: number of context examples per cluster.
- `top_n`: number of frequent forms/SEMCLASS values to show.
- `semclass_lemma_examples`: maximum lemma examples per SEMCLASS in a cluster.
- `random_state`: seed used when examples are sampled.

Returns: summary DataFrame.

### `run_clustering(...)`

Run one clustering config and build labels, scores, summary, and alignment.

Arguments:

- `embeddings`: 2D matrix.
- `tokens`: token DataFrame.
- `config`: `ClusterConfig`.
- `hierarchy`: optional hierarchy DataFrame or path.
- `hierarchy_depths`: hierarchy depths to evaluate.
- `show_progress`: show progress bars.

Returns: result dictionary with `labels`, `scores`, `summary`,
`hierarchy_alignment`, `model`, and `config`.

### `run_clustering_with_training_data(...)`

Fit on training data, label CoBaLD/evaluation data, and score evaluation rows.

Arguments:

- `training_embeddings`: fitting matrix.
- `evaluation_embeddings`: matrix to label.
- `evaluation_tokens`: token DataFrame aligned with evaluation embeddings.
- `config`: `ClusterConfig`.
- `hierarchy`: optional hierarchy DataFrame or path.
- `hierarchy_depths`: hierarchy depths to evaluate.
- `show_progress`: show progress bars.

Returns: result dictionary.

### `run_clustering_suite(...)`

Run several configs against an embedding payload.

Arguments:

- `embedding_payload`: payload dictionary or path to saved embedding pickle.
- `configs`: sequence of `ClusterConfig`; defaults to scalable defaults.
- `hierarchy`: optional hierarchy DataFrame or path.
- `hierarchy_depths`: hierarchy depths to evaluate.
- `show_progress`: show progress bars.

Returns: list of result dictionaries.

### `save_clustering_results(results, output_path)`

Save raw result dictionaries to pickle.

Arguments:

- `results`: sequence of result dictionaries.
- `output_path`: pickle path.

Returns: saved `Path`.

### `scores_to_dataframe(results)`

Convert result scores into one row per run.

Arguments:

- `results`: result dictionaries.

Returns: DataFrame.

## Result Export Helpers

Import:

```python
from cobaldclusterisation.cluster_exports import (
    ClusteringExportPaths,
    annotate_tokens,
    default_results_dir,
    output_prefix,
    semclass_cluster_map_table,
    write_clustering_outputs,
    write_conllu_plus,
)
```

### `ClusteringExportPaths(...)`

Dataclass returned by `write_clustering_outputs`.

Arguments:

- `excel`: summary workbook path.
- `semclass_cluster_map_excel`: semantic-class to automatic-cluster workbook
  path.
- `hierarchy_alignment_excel`: hierarchy alignment workbook path.
- `scores_csv`: score CSV path.
- `scores_xlsx`: score workbook path.
- `scores_txt`: per-run score text paths.
- `artifacts_pickle`: reusable result pickle path.
- `annotated_conllu_plus`: first CoNLL-U Plus export path, or `None`.
- `annotated_conllu_plus_files`: all CoNLL-U Plus export paths.

### `default_results_dir(output_dir, drive_dir, *, default_output_dir)`

Resolve where result files should be written.

Arguments:

- `output_dir`: requested working output directory.
- `drive_dir`: Drive output root.
- `default_output_dir`: default Colab output directory, normally `/content`.

Returns: `drive_dir/results` when `output_dir` equals the default, otherwise
`output_dir`.

### `output_prefix(results, *, label, family=None)`

Build a filename prefix.

Arguments:

- `results`: result dictionaries.
- `label`: model/run label.
- `family`: optional family override such as `Hierarchical`,
  `Minibatch_Kmeans`, or `Flat`.

Returns: safe filename prefix.

### `write_clustering_outputs(...)`

Write Drive-facing result artifacts.

Arguments:

- `results`: result dictionaries from clustering.
- `tokens`: CoBaLD target-token DataFrame aligned with result labels.
- `output_dir`: result directory.
- `label`: model/run label used in filenames.
- `family`: optional clustering family override.
- `metadata`: optional run metadata stored in the artifact pickle.

Returns: `ClusteringExportPaths`.

### `semclass_cluster_map_table(annotated_tokens, *, result)`

Build the semantic-class to automatic-cluster table used by
`{prefix}_semclass_cluster_map.xlsx`.

Arguments:

- `annotated_tokens`: token DataFrame containing `SEMCLASS` and numeric
  `cluster` columns, usually produced by `annotate_tokens`.
- `result`: clustering result dictionary containing `summary`; `cluster_name`
  values are used to format cluster names as `cluster_id:cluster_name`.

Returns: DataFrame with `semclass`, `automatic_clusters`, and `cluster_count`.

### `annotate_tokens(tokens, labels, *, result)`

Attach automatic cluster labels to a token DataFrame.

Arguments:

- `tokens`: token DataFrame.
- `labels`: cluster labels aligned with rows.
- `result`: result dictionary containing `summary` and `config`.

Returns: annotated DataFrame with `cluster` and `AUTO_SEMCLASS`.

### `write_conllu_plus(tokens, output_path)`

Write an annotated target-token CoNLL-U Plus file.

Arguments:

- `tokens`: DataFrame containing original CoBaLD columns and
  `AUTO_SEMCLASS`.
- `output_path`: output `.conllu` path.

Returns: output path as string.

## Score and Excel Helpers

Import:

```python
from cobaldclusterisation.scores import (
    format_score_table,
    format_score_value,
    metric_reference,
    print_result_scores,
    scores_to_metric_dataframe,
    write_scores,
)
from cobaldclusterisation.summary_excel import (
    write_cluster_summary_excel,
    write_hierarchy_alignment_excel,
    write_semclass_cluster_map_excel,
)
```

### `metric_reference(metric_name)`

Return a short interpretation note for a metric.

Arguments:

- `metric_name`: score name.

Returns: string.

### `format_score_value(metric_name, value)`

Format a metric value for text output.

Arguments:

- `metric_name`: score name.
- `value`: score value.

Returns: string.

### `format_score_table(title, scores)`

Build a printable score table.

Arguments:

- `title`: table title.
- `scores`: score dictionary.

Returns: string.

### `print_result_scores(label, result)`

Print and return one formatted score table.

Arguments:

- `label`: run label.
- `result`: clustering result dictionary.

Returns: printed table string.

### `scores_to_metric_dataframe(results)`

Convert scores into a metric-row comparison table.

Arguments:

- `results`: result dictionaries.

Returns: DataFrame.

### `write_scores(results, *, output_dir, label)`

Write score CSV, XLSX, and per-run TXT files.

Arguments:

- `results`: result dictionaries.
- `output_dir`: output directory.
- `label`: filename label.

Returns: `(scores_csv, scores_xlsx, scores_txt_paths)`.

### `write_cluster_summary_excel(results, output_path)`

Write all cluster summaries to one workbook.

Arguments:

- `results`: result dictionaries containing `summary` and `config`.
- `output_path`: workbook path.

Returns: output path string.

### `write_semclass_cluster_map_excel(tables, output_path)`

Write semantic-class to automatic-cluster map sheets to one workbook.

Arguments:

- `tables`: sequence of `(result, table)` pairs, where each table is usually
  produced by `semclass_cluster_map_table`.
- `output_path`: workbook path.

Returns: output path string.

### `write_hierarchy_alignment_excel(results, output_path)`

Write hierarchy alignment sheets to one workbook.

Arguments:

- `results`: result dictionaries containing `hierarchy_alignment` and `config`.
- `output_path`: workbook path.

Returns: output path string.

## SEMCLASS Statistics

Import:

```python
from cobaldclusterisation.semclass_stats import (
    build_semclass_threshold_table,
    hierarchy_semclass_count,
    semclass_counts,
    semclass_table_to_markdown,
    semclass_threshold_table_from_corpus,
)
```

### `semclass_counts(tokens)`

Count valid SEMCLASS values.

Arguments:

- `tokens`: token DataFrame.

Returns: `Series`.

### `build_semclass_threshold_table(counts, *, max_occurrences=15, existing_semclass_count=None)`

Build cumulative SEMCLASS threshold statistics.

Arguments:

- `counts`: SEMCLASS count series.
- `max_occurrences`: maximum occurrence threshold.
- `existing_semclass_count`: optional total class count from a hierarchy.

Returns: DataFrame.

### `hierarchy_semclass_count(hierarchy)`

Count classes listed in a hierarchy CSV.

Arguments:

- `hierarchy`: hierarchy CSV path.

Returns: integer.

### `semclass_threshold_table_from_corpus(...)`

Load a corpus and build the SEMCLASS threshold table.

Arguments:

- `data_dir`: CoBaLD corpus directory.
- `splits`: split names to load.
- `max_occurrences`: maximum threshold.
- `hierarchy`: optional hierarchy CSV path.

Returns: DataFrame.

### `semclass_table_to_markdown(table)`

Convert a threshold table to Markdown.

Arguments:

- `table`: threshold DataFrame.

Returns: Markdown string.

## Hierarchical Baseline Pipelines

The main hierarchical pipelines are:

```python
from cobaldclusterisation.rubert_baseline import run as run_rubert
from cobaldclusterisation.sambalingo_baseline import run as run_sambalingo
from cobaldclusterisation.gigachat_baseline import run as run_gigachat
```

All three return a dictionary with:

- `results`: clustering result dictionaries.
- `paths`: dataclass-as-dict containing embeddings, clustering features where
  relevant, summary/scores/alignment files, artifact pickle, and CoNLL-U Plus
  exports.
- `training_source`, `training_data_paths`, and payload dictionaries.

### Common hierarchical `run(...)` arguments

These arguments are shared by ruBERT, SambaLingo, and GigaChat unless noted.

- `data_dir`: CoBaLD corpus directory.
- `hierarchy`: hierarchy DataFrame/path, or `None` to skip hierarchy scores.
- `splits`: CoBaLD split files to load.
- `algorithms`: algorithm names. Common values: `BisectingKMeans`, `BIRCH`,
  `KNNLeiden`, `KMeans`, `MiniBatchKMeans`, `Agglomerative`, `HDBSCAN`.
- `n_clusters`: integer, sequence of integers, or `data_semclass`.
- `output_dir`: local working output directory.
- `results_dir`: explicit result directory. If omitted and
  `output_dir="/content"`, results go to `drive_dir/results`.
- `save_embeddings_to_drive`: save embedding pickles to `drive_dir`.
- `drive_dir`: Drive root, normally `/content/drive/MyDrive/cobald_outputs`.
- `embedding_filename`: filename for full CoBaLD embeddings.
- `embeddings_path`: existing CoBaLD embedding pickle to reuse.
- `training_source`: `cobald`, `syntagrus`, or `syntagrus_cobald`.
- `training_embedding_filename`: filename for generated training embeddings.
- `training_embeddings_path`: existing training embedding pickle to reuse.
- `syntagrus_dir`: UD Russian SynTagRus directory or file.
- `syntagrus_files`: SynTagRus filenames to load from `syntagrus_dir`.
- `syntagrus_repo_url`: Git URL for SynTagRus.
- `clone_syntagrus_if_missing`: clone SynTagRus when missing.
- `excel_filename`: legacy argument; result workbook names are now generated
  from the output prefix.
- `hierarchy_alignment_filename`: legacy argument; result workbook names are
  now generated from the output prefix.
- `label`: model/run label used in output prefixes.
- `embedding_batch_size`: transformer batch size.
- `clustering_batch_size`: batch size for supported clustering algorithms.
- `max_length`: tokenizer maximum length.
- `device`: `cuda`, `cpu`, or `None`.
- `seed`: random seed.
- `include_punctuation_context`: keep punctuation in transformer context.
- `normalize_embeddings`: normalize raw embeddings.
- `normalize_for_clustering`: normalize vectors before clustering.
- `n_init`: initialization count for KMeans-like algorithms.
- `min_samples`: HDBSCAN minimum samples.
- `min_cluster_size`: HDBSCAN minimum cluster size.
- `hdbscan_cluster_selection_epsilon`: HDBSCAN epsilon.
- `hdbscan_cluster_selection_method`: HDBSCAN selection method.
- `hdbscan_allow_single_cluster`: HDBSCAN option.
- `hdbscan_n_jobs`: HDBSCAN worker count.
- `metric`: metric for supported algorithms.
- `birch_threshold`: BIRCH threshold.
- `birch_branching_factor`: BIRCH branching factor.
- `bisecting_strategy`: BisectingKMeans split strategy.
- `graph_n_neighbors`: kNN graph neighbor count.
- `graph_resolution`: Leiden/Louvain graph resolution.
- `graph_metric`: kNN graph metric.
- `graph_n_jobs`: kNN graph worker count.
- `hierarchy_depths`: hierarchy depths to score.
- `show_progress`: show progress bars.

### `rubert_baseline.run(...)`

ruBERT-specific defaults:

- `model_name` is fixed internally to `cointegrated/rubert-tiny2`.
- `embedding_batch_size=32`.
- `normalize_for_clustering=True`.
- No PCA projection is applied by default.

### `rubert_baseline.build_cluster_configs(...)`

Build `ClusterConfig` objects from Colab-friendly algorithm names.

Arguments:

- `algorithms`: sequence of names.
- `n_clusters`: integer, sequence, or `data_semclass`.
- `tokens`: token DataFrame needed when `n_clusters="data_semclass"`.
- Remaining arguments map directly to `ClusterConfig` fields.

Returns: list of `ClusterConfig`.

### `rubert_baseline.prepare_training_and_cobald_payloads(...)`

Prepare CoBaLD and optional SynTagRus training embedding payloads. This is
mostly useful for custom notebooks that want to separate embedding generation
from clustering.

Arguments:

- `data_dir`, `splits`: CoBaLD corpus input.
- `training_source`: `cobald`, `syntagrus`, or `syntagrus_cobald`.
- `output_dir`, `embedding_output_path`, `training_embedding_filename`:
  output paths.
- `embeddings_path`, `training_embeddings_path`: reuse saved payloads.
- `drive_output_dir`: optional Drive directory for generated embeddings.
- `embedding_config`: `EmbeddingConfig`.
- `syntagrus_dir`, `syntagrus_files`, `syntagrus_repo_url`,
  `clone_syntagrus_if_missing`: SynTagRus inputs.
- `show_progress`: progress bars.
- `generate_cobald_embeddings`: injectable embedding function.

Returns: `(cobald_payload, training_payload, training_source, data_paths)`.

### `sambalingo_baseline.run(...)`

SambaLingo-specific arguments in addition to common hierarchical arguments:

- `clustering_features_filename`: filename for reduced CoBaLD clustering
  features. Defaults to a PCA-derived name.
- `training_clustering_features_filename`: filename for reduced training
  features when training source is not CoBaLD.
- `torch_dtype`: default `float16`.
- `projection_n_components`: PCA dimension for clustering features.
- `projection_batch_size`: IncrementalPCA batch size.
- `normalize_projection_input`: normalize vectors before PCA fitting/transform.
- `normalize_projection_output`: normalize reduced vectors.
- `agglomerative_max_rows`: row guard for agglomerative clustering.
- `allow_quadratic_algorithms`: override the agglomerative guard.
- `release_cuda_after_embeddings`: clear CUDA cache before PCA/clustering.

### `sambalingo_baseline.reduce_embeddings_incremental_pca(...)`

Reduce an embedding matrix with IncrementalPCA.

Arguments:

- `embeddings`: 2D matrix.
- `n_components`: reduced dimension.
- `batch_size`: IncrementalPCA batch size.
- `normalize_input`: normalize before PCA.
- `normalize_output`: normalize reduced vectors.
- `show_progress`: progress bars.

Returns: reduced matrix.

### `sambalingo_baseline.make_clustering_payload(...)`

Build a reduced clustering payload from a full embedding payload.

Arguments:

- `embedding_payload`: payload with `embeddings` and `tokens`.
- `projection_n_components`: PCA dimension.
- `projection_batch_size`: PCA batch size.
- `normalize_projection_input`: normalize before PCA.
- `normalize_projection_output`: normalize after PCA.
- `show_progress`: progress bars.

Returns: payload dictionary.

### `sambalingo_baseline.make_training_and_cobald_clustering_payloads(...)`

Fit PCA on training embeddings and transform training plus CoBaLD embeddings.

Arguments:

- `training_payload`: training embedding payload.
- `cobald_payload`: CoBaLD embedding payload.
- `projection_n_components`: PCA dimension.
- `projection_batch_size`: PCA batch size.
- `normalize_projection_input`: normalize before PCA.
- `normalize_projection_output`: normalize after PCA.
- `show_progress`: progress bars.

Returns: `(training_clustering_payload, cobald_clustering_payload)`.

### `gigachat_baseline.run(...)`

GigaChat-specific arguments in addition to common hierarchical arguments:

- `clustering_features_filename`: filename for reduced CoBaLD clustering
  features. Defaults to a PCA-derived name.
- `training_clustering_features_filename`: filename for reduced training
  features when training source is not CoBaLD.
- `torch_dtype`: default `bfloat16`.
- `trust_remote_code`: passed to Hugging Face model/tokenizer loading.
- `projection_n_components`, `projection_batch_size`,
  `normalize_projection_input`, `normalize_projection_output`: PCA controls.
- `agglomerative_max_rows`, `allow_quadratic_algorithms`: agglomerative guard.
- `release_cuda_after_embeddings`: clear CUDA cache before PCA/clustering.

## External MiniBatch K-Means Pipeline

Import:

```python
from cobaldclusterisation.external_minibatch_baseline import (
    RawTextChunk,
    collect_external_contexts,
    collect_ud_conllu_contexts,
    fit_minibatch_kmeans_incremental,
    is_word_token,
    iter_raw_text_chunks,
    iter_ud_conllu_chunks,
    load_external_dataset_stream,
    make_tokenizer_safe_external_chunks,
    run,
    run_gigachat_external,
    run_gigachat_syntagrus_external,
    run_rubert_external,
    run_rubert_syntagrus_external,
    run_sambalingo_external,
    run_sambalingo_syntagrus_external,
    split_chunk_for_tokenizer,
    tokenize_raw_text,
)
```

### `RawTextChunk(tokens, target_indices, document_index, chunk_index, lemmas=None)`

Dataclass for external embedding contexts.

Arguments:

- `tokens`: pretokenized context tokens.
- `target_indices`: indices within `tokens` to embed.
- `document_index`: source document number.
- `chunk_index`: chunk number.
- `lemmas`: optional lemma list aligned with tokens.

### `split_chunk_for_tokenizer(chunk, *, tokenizer, max_length)`

Split one chunk so tokenized pieces fit a model limit.

Arguments:

- `chunk`: `RawTextChunk`.
- `tokenizer`: Hugging Face fast tokenizer.
- `max_length`: model maximum length.

Returns: list of `RawTextChunk`.

### `is_word_token(token)`

Return whether a raw token is word-like.

Arguments:

- `token`: string token.

Returns: bool.

### `tokenize_raw_text(text)`

Tokenize raw text into simple word/punctuation tokens.

Arguments:

- `text`: raw document text.

Returns: list of strings.

### `iter_raw_text_chunks(...)`

Yield external raw-text chunks.

Arguments:

- `text`: raw document text.
- `document_index`: source document number.
- `start_chunk_index`: first chunk index.
- `max_context_tokens`: maximum total context tokens per chunk.
- `max_context_word_tokens`: maximum word-like target tokens per chunk.

Returns: iterator of `RawTextChunk`.

### `iter_ud_conllu_chunks(...)`

Yield SynTagRus/UD chunks.

Arguments:

- `paths`: UD `.conllu` paths.
- `max_context_tokens`: maximum total context tokens per chunk.
- `max_context_word_tokens`: maximum word-like target tokens per chunk.
- `max_sentences`: optional sentence cap.

Returns: iterator of `RawTextChunk`.

### `collect_ud_conllu_contexts(...)`

Collect UD chunks up to a token budget.

Arguments:

- `paths`: UD `.conllu` paths.
- `max_train_tokens`: maximum target tokens to collect.
- `max_sentences`: optional sentence cap.
- `max_context_tokens`: maximum context tokens per chunk.
- `max_context_word_tokens`: maximum word targets per chunk.
- `show_progress`: progress bars.

Returns: list of `RawTextChunk`.

### `collect_external_contexts(...)`

Collect raw-text chunks from a streaming dataset.

Arguments:

- `dataset`: iterable of document dictionaries.
- `max_train_tokens`: maximum target tokens.
- `text_column`: document text field.
- `max_documents`: optional document cap.
- `max_context_tokens`: maximum context tokens per chunk.
- `max_context_word_tokens`: maximum word targets per chunk.
- `show_progress`: progress bars.

Returns: list of `RawTextChunk`.

### `make_tokenizer_safe_external_chunks(...)`

Split chunks so they are safe for a tokenizer/model length.

Arguments:

- `chunks`: raw chunks.
- `tokenizer`: Hugging Face tokenizer.
- `max_length`: model maximum length.
- `show_progress`: progress bars.

Returns: list of `RawTextChunk`.

### `load_external_dataset_stream(...)`

Load a streaming Hugging Face dataset.

Arguments:

- `dataset_name`: dataset id.
- `dataset_subset`: dataset subset/config.
- `split`: split name.
- `seed`: shuffle seed.
- `shuffle_buffer_size`: streaming shuffle buffer size.

Returns: iterable of dictionaries.

### `fit_minibatch_kmeans_incremental(...)`

Fit MiniBatchKMeans on a possibly memmapped feature matrix.

Arguments:

- `features`: 2D matrix or memmap.
- `n_clusters`: cluster count.
- `batch_size`: MiniBatchKMeans batch size.
- `random_state`: seed.
- `n_init`: initialization count.
- `normalize`: normalize feature batches before fitting.
- `show_progress`: progress bars.

Returns: fitted `MiniBatchKMeans`.

### `external_minibatch_baseline.run(...)`

Train external MiniBatch K-Means and predict CoBaLD clusters.

Important arguments:

- `data_dir`, `hierarchy`, `splits`: CoBaLD input and hierarchy.
- `external_source`: `fineweb2` or `syntagrus_ud`.
- `dataset_name`, `dataset_subset`, `split`, `text_column`: FineWeb2 stream
  settings.
- `syntagrus_dir`, `syntagrus_files`, `syntagrus_repo_url`,
  `clone_syntagrus_if_missing`: SynTagRus source settings.
- `max_train_tokens`: external target-token budget.
- `max_documents`: document or sentence cap.
- `shuffle_buffer_size`: FineWeb2 stream shuffle buffer.
- `output_dir`: local working directory.
- `results_dir`: explicit results directory.
- `save_models_to_drive`: save model/features to `drive_dir`.
- `drive_dir`: Drive root.
- `model_name`: Hugging Face encoder model.
- `model_label`: output label used in filenames.
- `n_clusters`: integer, sequence, or `data_semclass`.
- `embedding_batch_size`: transformer batch size.
- `kmeans_batch_size`: KMeans fitting/prediction batch size.
- `max_length`: tokenizer max length.
- `device`, `torch_dtype`, `trust_remote_code`: model loading controls.
- `seed`: random seed.
- `include_punctuation_context`: keep punctuation in CoBaLD contexts.
- `normalize_embeddings`: normalize raw hidden-state embeddings.
- `normalize_for_clustering`: normalize feature batches for KMeans.
- `projection_n_components`: optional IncrementalPCA dimension.
- `projection_batch_size`: PCA batch size.
- `normalize_projection_input`: normalize before PCA.
- `normalize_projection_output`: normalize after PCA.
- `max_context_tokens`: external context length in tokens.
- `max_context_word_tokens`: external target-token count per context.
- `n_init`: MiniBatchKMeans initialization count.
- `hierarchy_depths`: hierarchy depths to evaluate.
- `show_progress`: progress bars.

Returns: dictionary with `results`, `paths`, and `config`.

### External convenience wrappers

The following functions call `external_minibatch_baseline.run(**kwargs)` with
model/source defaults. Any keyword accepted by `run` can be passed through.

- `run_rubert_external(**kwargs)`: ruBERT base model, FineWeb2 source.
- `run_rubert_syntagrus_external(**kwargs)`: ruBERT base model, SynTagRus
  source.
- `run_sambalingo_external(**kwargs)`: SambaLingo model, FineWeb2 source, PCA
  defaults.
- `run_sambalingo_syntagrus_external(**kwargs)`: SambaLingo model, SynTagRus
  source, PCA defaults.
- `run_gigachat_external(**kwargs)`: GigaChat model, FineWeb2 source, PCA
  defaults.
- `run_gigachat_syntagrus_external(**kwargs)`: GigaChat model, SynTagRus
  source, PCA defaults.

## Baseline Payload Helper

Import:

```python
from cobaldclusterisation.baseline_payloads import load_saved_embedding_payload
```

### `load_saved_embedding_payload(path)`

Load a saved embedding payload and validate that it has the required baseline
fields.

Arguments:

- `path`: pickle path.

Returns: payload dictionary with `saved_path`.
