# Datasets and Large Payloads

Kaggle kernels have strict code size limits (approximately 5 MB for the main script and inline payload). `kagglex` provides built-in mechanisms to attach existing datasets, create datasets, and automatically offload large payloads.

## Attaching Existing Kaggle Datasets

Attach public or private Kaggle datasets to your run using `--dataset`:

```bash
kagglex run \
  --file train.py \
  --dataset "username/imdb-reviews" \
  --dataset "stanford/stanford-sentiment-treebank"
```

Attached datasets are mounted read-only at `/kaggle/input/<dataset-slug>/`.

## Creating and Updating Datasets

You can create or update Kaggle datasets using `kagglex dataset push`:

```bash
kagglex dataset push \
  --dir ./data/processed \
  --title "Processed Tokenized Corpus" \
  --slug "processed-tokenized-corpus"
```

By default, datasets are created as private. Pass `--public` to make them public:

```bash
kagglex dataset push \
  --dir ./data/benchmark \
  --title "Public Benchmark Set" \
  --public
```

## Automated Large Payload Offloading

When your project repository or bundled local data exceeds Kaggle's inline size limit, use the `--auto-dataset` flag:

```bash
kagglex run \
  --dir . \
  --command "python train.py" \
  --include-data ./embeddings \
  --auto-dataset
```

### How Auto-Dataset Works

1. `kagglex` detects the total packaged size of your project and local data.
2. If `--auto-dataset` is enabled, `kagglex` publishes a private Kaggle dataset containing your packaged source code and data files under the slug `kagglex-payload-<run-slug>`.
3. The dataset is automatically attached to the kernel's metadata.
4. During execution, the remote bootstrap script mounts the dataset payload from `/kaggle/input/` and unpacks it into `/kaggle/working`.
5. Subsequent runs with the same slug version the payload dataset seamlessly.
