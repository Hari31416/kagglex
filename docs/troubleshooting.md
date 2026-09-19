# Troubleshooting

Solutions and diagnostic steps for common issues encountered when using `kagglex`.

## Authentication Errors

### `Unauthorized: Kaggle credentials not found`

**Problem**: The Kaggle API client cannot find authentication credentials.

**Solution**:
1. Ensure your token exists at `~/.kaggle/kaggle.json`.
2. Check permissions on Unix/macOS:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Alternatively, set environment variables:
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```

### `403 Forbidden: Phone Verification Required`

**Problem**: Kaggle requires phone number verification before permitting GPU and TPU accelerator access.

**Solution**:
1. Log in to [kaggle.com](https://www.kaggle.com).
2. Go to **Account Settings** -> **Phone Verification**.
3. Complete SMS verification.

## Size Limit Errors

### `Compressed archive exceeds Kaggle 5MB inline limit`

**Problem**: The project repository or bundled local files exceed 5 MB.

**Solution**:
1. Create a `.kaggleignore` file to exclude large folders (e.g. `data/`, `checkpoints/`, `.venv/`).
2. Add the `--auto-dataset` flag to automatically offload payloads to a private Kaggle dataset:
   ```bash
   kagglex run --file train.py --auto-dataset
   ```

## Interactive REPL Connection Issues

### `Failed to connect to Kaggle Jupyter proxy`

**Problem**: `kagglex exec` cannot reach the Jupyter proxy URL.

**Solution**:
1. Confirm the interactive Kaggle notebook session is actively running in your browser.
2. Ensure the full URL including the `token=` parameter was copied:
   ```bash
   kagglex exec --url "https://kkb-production.jupyter-proxy.kaggle.net/k/.../?token=..." --test
   ```
3. Proxy URLs expire when a notebook session times out or is stopped.

## GPU and TPU Quotas

### `Kaggle GPU quota exceeded`

**Problem**: Kaggle limits GPU compute to 30 hours per rolling 7 days.

**Solution**:
1. Check your consumption breakdown:
   ```bash
   kagglex quota --days 7
   ```
2. Switch to TPU compute (`--gpu v3-8`, which has a separate 20-hour weekly quota) or standard CPU (`--gpu none`).
3. Kaggle weekly quotas roll over continuously 7 days after each run's timestamp.
