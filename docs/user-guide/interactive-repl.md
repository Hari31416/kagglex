# Interactive REPL Sessions

The `kagglex exec` command allows you to connect directly to an active Kaggle notebook session via the Kaggle Jupyter Proxy server. This provides an interactive workflow with sub-second feedback without waiting for batch kernel scheduling.

## Getting the Proxy URL

1. Open or start an interactive notebook in your browser at [kaggle.com/code](https://www.kaggle.com/code).
2. Start the notebook session and select your desired accelerator (e.g. GPU T4 x2).
3. In the top navigation menu, click **Run** -> **Kaggle Jupyter Server** -> **Copy URL**.
4. The copied URL will look similar to:

```text
https://kkb-production.jupyter-proxy.kaggle.net/k/12345678/abcdef123456?token=kaggle-proxy-token
```

## Setting the Proxy URL

You can pass the URL explicitly using `--url`, configure it via the `KAGGLE_JUPYTER_URL` environment variable, or define it in configuration files.

```bash
export KAGGLE_JUPYTER_URL="https://kkb-production.jupyter-proxy.kaggle.net/k/12345678/abcdef123456?token=kaggle-proxy-token"
```

## Testing Connection

Verify connectivity to the running Kaggle kernel:

```bash
kagglex exec --test
```

## GPU Profiling

Query GPU device count, device names, and VRAM memory utilization directly:

```bash
kagglex exec --gpu-info
```

## Executing Python Code Remotely

### Inline Code Snippets

Run arbitrary Python code directly from your terminal:

```bash
kagglex exec "import torch; print('CUDA Devices:', torch.cuda.device_count())"
```

### Local Python Files

Execute a local script on the remote GPU session without staging or uploading a new kernel:

```bash
kagglex exec --file test_inference.py
```

## Bidirectional File Transfers

You can transfer files directly to and from `/kaggle/working` in the live notebook environment:

### List Remote Files

```bash
kagglex exec --list-files
```

### Uploading Local Files

```bash
kagglex exec --upload ./checkpoint_epoch_5.pt
```

### Downloading Remote Files

```bash
kagglex exec --download predictions.csv -o ./local_predictions.csv
```
