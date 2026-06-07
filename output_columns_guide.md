# Output Files Guide for Human Review

This guide explains the files meant for linguistic inspection. The automatic
clusters should be treated as evidence for review, not as final semantic
classes.

## SUMMARIES

File pattern: `{prefix}_summary.xlsx`

Each sheet corresponds to one clustering run. Sheet names include the algorithm
and run setting, for example `0_birch_k100`.

`cluster`: Numeric automatic cluster id.

`cluster_name`: A short readable name for the cluster. It is the lemma, or form
when the lemma is unavailable, of the token closest to the cluster centroid.
This name is only a convenience label, not a human-approved semantic class.

`token_count`: Number of CoBaLD target tokens assigned to this automatic
cluster.

`lemma_count`: Number of distinct non-empty lemmas in the cluster.

`lemmas`: Lemmas found in the cluster, with counts. Values are newline
separated inside the Excel cell.

`top_forms`: Most frequent surface forms in the cluster, with counts.

`semclasses`: Manual CoBaLD `SEMCLASS` labels represented in the cluster, with
counts. This is useful for seeing which gold semantic classes the automatic
cluster overlaps.

`semclass_lemma_examples`: Up to 50 lemma examples for each manual semantic
class represented in the automatic cluster. This is intended for quick
linguistic inspection of what kinds of lemmas connect a cluster to a manual
class.

`examples`: A few token examples with their manual `SEMCLASS` and sentence
context. These examples are for orientation only.

## SEMCLASS CLUSTER MAP

File pattern: `{prefix}_semclass_cluster_map.xlsx`

Each sheet corresponds to one clustering run. This workbook starts from the
manual semantic classes and shows which automatic clusters contain examples of
each class.

`semclass`: Manual CoBaLD `SEMCLASS` label.

`automatic_clusters`: Automatic clusters that contain at least one token with
this manual semantic class. Values are newline separated inside the Excel cell.
Each value has the form `cluster_id:cluster_name`, for example `0:банк`.
`cluster_name` is the automatic readable cluster name derived from the token
closest to the cluster centroid; it is not a human-approved class name.

`cluster_count`: Number of distinct automatic clusters listed in
`automatic_clusters`. For example, if `BEING` appears in clusters `0`, `1`,
and `2`, this value is `3`.

## SCORES

File patterns: `{prefix}_scores.xlsx`, `{prefix}_scores.csv`,
`{prefix}_*_scores.txt`

The XLSX and CSV score tables have one row per metric and one column per
clustering run.

`metric`: Name of the evaluation metric.

`reference_note`: Short interpretation note, for example whether higher or
lower values are better.

`run_...`: Score value for one clustering run. The column name includes the
run number, algorithm, and run setting.

Common metric names:

`silhouette`: Internal cluster separation score. Range is -1 to 1; higher is
better.

`calinski_harabasz`: Internal cluster separation/compactness score. Higher is
better.

`davies_bouldin`: Internal cluster overlap score. Lower is better.

`n_clusters_found`: Number of clusters actually present in the output labels.

`n_noise`: Number of tokens assigned to noise by algorithms that can produce
noise labels, such as HDBSCAN.

`semclass_exact_adjusted_rand`: Adjusted Rand Index against exact manual
`SEMCLASS` labels. Higher is better; 0 is near random.

`semclass_exact_normalized_mutual_info`: Normalized mutual information against
exact manual `SEMCLASS` labels. Higher is better.

`semclass_exact_homogeneity`: Whether automatic clusters avoid mixing manual
classes. Higher is better.

`semclass_exact_completeness`: Whether each manual class is mostly contained
inside one automatic cluster. Higher is better.

`semclass_exact_v_measure`: Harmonic mean of homogeneity and completeness.
Higher is better.

`semclass_exact_purity`: For each automatic cluster, the share explained by its
most frequent manual class, averaged by tokens. Higher is better, but it can be
inflated by many small clusters.

`semclass_exact_n_labeled`: Number of tokens with usable manual `SEMCLASS`
labels included in the exact-label score.

Metrics with names like `hierarchy_depth_2_*` repeat the same external score
logic after replacing exact `SEMCLASS` labels with their ancestor class at that
semantic hierarchy depth.

The TXT score files contain the same scores in a plain text format suitable for
copying into notes.

## HIERARCHY ALIGNMENT

File pattern: `{prefix}_hierarchy_alignment.xlsx`

Each sheet corresponds to one clustering run. Rows show how each automatic
cluster aligns to the human semantic hierarchy at each requested depth.

`depth`: Hierarchy depth being evaluated.

`cluster`: Numeric automatic cluster id.

`cluster_token_count`: Total number of CoBaLD target tokens in this automatic
cluster.

`cluster_labeled_count`: Number of tokens in the cluster that have a usable
manual hierarchy label at this depth.

`best_label`: The most frequent manual hierarchy label in this automatic
cluster at this depth.

`best_label_count`: Number of cluster tokens with `best_label`.

`cluster_purity`: `best_label_count / cluster_labeled_count`. This answers:
within this automatic cluster, how dominant is the best matching manual label?

`gold_label_coverage`: Share of all tokens with this manual hierarchy label
that are captured by this automatic cluster. This answers: how much of the
manual class did this automatic cluster recover?

## CLUSTERED TOKEN TABLES

File patterns, mainly for the external MiniBatch K-Means pipeline:
`{model_label}_cobald_clustered_tokens*.xlsx` and `.csv`

These are compatibility tables containing one row per CoBaLD target token and a
predicted numeric cluster column.

`cluster`: Numeric automatic cluster id.

`ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, `DEPS`,
`MISC`, `DEEPSLOT`, `SEMCLASS`: Original CoBaLD token columns. `SEMCLASS` is
the manual annotation.

`sentence_id`: Sentence id from the original CoNLL-U comments, when available.

`split`: Corpus split, usually `train` or `dev`.

`sentence_index`: Numeric sentence position inside the loaded split.

`token_index`: Numeric token-row position inside the sentence.

`is_punctuation`: Whether the original token is punctuation. In target-token
outputs this should normally be false.

`is_empty_node`: Whether the original row is an empty/ellipsis row. In
target-token outputs this should normally be false.

`context_text`: Sentence text used for examples and inspection.

`embedding_context_text`: Tokenized context used for embedding extraction when
available. Depending on the run settings, this context may include punctuation
even though punctuation itself is not clustered.

`embedding_index`: Row index aligning this token with the embedding matrix.

Some runs may include only a subset of these metadata columns depending on how
the embedding payload was produced.

## ANNOTATED CONLL-U PLUS

File pattern: `{prefix}_{run}_annotated.conllu`

These files are intended for human reading and corpus-style lookup. They use a
CoNLL-U Plus-style header:

```text
# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC DEEPSLOT SEMCLASS AUTO_SEMCLASS
```

The file contains one row per clustered CoBaLD target token. It does not
contain punctuation marks, multiword-token range rows, or empty/ellipsis rows.
This is intentional: the embedding and clustering pipelines assign automatic
semantic classes only to surface non-punctuation target tokens. Punctuation may
still have been present in the transformer context, but punctuation itself has
no embedding row and therefore no automatic cluster label.

Comment lines:

`# sent_id = ...`: Original sentence id, when available.

`# split = ...`: Corpus split, usually `train` or `dev`.

Token columns:

`ID`: Token id from the original sentence.

`FORM`: Surface word form.

`LEMMA`: Lemma.

`UPOS`: Universal part of speech.

`XPOS`: Language- or corpus-specific part-of-speech tag.

`FEATS`: Morphological features.

`HEAD`: Syntactic head id.

`DEPREL`: Dependency relation.

`DEPS`: Enhanced dependencies.

`MISC`: Miscellaneous CoNLL-U information.

`DEEPSLOT`: Original CoBaLD deep semantic slot column.

`SEMCLASS`: Original manual CoBaLD semantic class. This is the human annotation
used for evaluation.

`AUTO_SEMCLASS`: Automatic cluster label assigned by the clustering pipeline.
It has the form `cluster_<number>:<cluster_name>`, for example
`cluster_12:банк`. The number is the internal automatic cluster id. The text
after the colon is a readable name derived from the representative lemma or
form nearest the cluster centroid. This last column is not a manual class and
should be interpreted as a machine-generated candidate grouping.

## ARTIFACT PICKLE

File pattern: `{prefix}_artifacts.pkl`

This file is for reuse in Python notebooks, not direct manual reading. It
contains the same information needed to regenerate summaries and scores without
rerunning clustering:

`format_version`: Internal format version.

`label`: Model/run label used in filenames.

`prefix`: Full output prefix.

`metadata`: Run metadata such as model name, training source, and hierarchy
depths.

`runs`: List of clustering runs. Each run contains `config`, `labels`,
`annotated_tokens`, `summary`, `scores`, and `hierarchy_alignment`.
