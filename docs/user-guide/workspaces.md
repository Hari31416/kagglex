# Workspaces and Packaging

`kagglex` inspects your local repository layout to package dependencies, source code, and local data cleanly before uploading to Kaggle.

## Project Layout Detection

`kagglex` automatically detects your project structure:

- **Standalone Scripts**: When pointing to a single `.py` file (e.g. `kagglex run --file script.py`), `kagglex` packages only the targeted script and required staging components.
- **Python Packages and Modules**: When pointing to a repository containing `src/`, `pyproject.toml`, or `setup.py`, `kagglex` packages the entire repository so relative imports and package submodules function identically in the cloud environment.

## Ignoring Files with .kaggleignore

To prevent uploading bulky artifacts, caches, virtual environments, or sensitive secrets, `kagglex` supports `.kaggleignore`.

Create a `.kaggleignore` file in your repository root using standard `.gitignore` syntax:

```text
# Ignore caches and environments
__pycache__/
*.py[cod]
.venv/
.git/

# Ignore large local datasets (use --auto-dataset or Kaggle datasets instead)
data/raw/
*.tar.gz
*.bin
*.pt

# Ignore local output artifacts
outputs/
results/
```

### Default Ignore Rules

Even without a `.kaggleignore` file, `kagglex` automatically ignores standard directories such as `.git`, `.venv`, `__pycache__`, and staging directories.

## Bundling Local Data

If your model requires local configuration files, tokenizers, or small test sets, bundle them using the `--include-data` argument:

```bash
kagglex run \
  --file train.py \
  --include-data ./configs \
  --include-data ./tokenizer.json
```

Included directories are packaged alongside the execution bootstrap script and extracted into `/kaggle/working` at runtime.
