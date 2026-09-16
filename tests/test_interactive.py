"""Unit tests for interactive Jupyter Proxy client."""

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kagglex.config import InteractiveConfig, resolve_jupyter_url
from kagglex.interactive import JupyterProxyClient


def test_url_resolution_env_and_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolution of jupyter URL from explicit arg or env var."""
    monkeypatch.delenv("KAGGLE_JUPYTER_URL", raising=False)
    assert resolve_jupyter_url(None) is None
    assert (
        resolve_jupyter_url("https://direct.proxy/token")
        == "https://direct.proxy/token"
    )

    monkeypatch.setenv("KAGGLE_JUPYTER_URL", "https://env.proxy?token=abc")
    assert resolve_jupyter_url(None) == "https://env.proxy?token=abc"
    assert resolve_jupyter_url("https://override.proxy") == "https://override.proxy"


def test_interactive_config() -> None:
    """Test InteractiveConfig dataclass."""
    cfg = InteractiveConfig(url="https://proxy.test?token=xyz", timeout=60)
    assert cfg.url == "https://proxy.test?token=xyz"
    assert cfg.timeout == 60


def test_client_init_query_format() -> None:
    """Test client initialization with query string token."""
    client = JupyterProxyClient(
        "https://kkb-production.jupyter-proxy.kaggle.net?token=secretjwt123"
    )
    assert client.base_url == "https://kkb-production.jupyter-proxy.kaggle.net"
    assert client._token == "secretjwt123"
    assert client._url_format == "query"

    rest_url = client._build_url("/api/kernels")
    assert (
        rest_url
        == "https://kkb-production.jupyter-proxy.kaggle.net/api/kernels?token=secretjwt123"
    )

    ws_url = client._build_ws_url("/api/kernels/k123/channels")
    assert (
        ws_url
        == "wss://kkb-production.jupyter-proxy.kaggle.net/api/kernels/k123/channels?token=secretjwt123"
    )


def test_client_init_path_format() -> None:
    """Test client initialization with path token format."""
    client = JupyterProxyClient("https://hub.kaggle.com/k/123456/jwt/proxy")
    assert client.base_url == "https://hub.kaggle.com/k/123456/jwt/proxy"
    assert client._token is None
    assert client._url_format == "path"

    rest_url = client._build_url("/api/kernels")
    assert rest_url == "https://hub.kaggle.com/k/123456/jwt/proxy/api/kernels"

    ws_url = client._build_ws_url("/api/kernels/k123/channels")
    assert ws_url == "wss://hub.kaggle.com/k/123456/jwt/proxy/api/kernels/k123/channels"


def test_test_connection_success() -> None:
    """Test test_connection when server responds with 200."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert client.test_connection() is True


def test_test_connection_failure() -> None:
    """Test test_connection when server fails or times out."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "get") as mock_get:
        mock_get.side_effect = requests.RequestException("Connection refused")
        assert client.test_connection() is False


def test_get_or_create_kernel_existing() -> None:
    """Test get_or_create_kernel when a running kernel already exists."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: [{"id": "kernel-abc-123"}]
        )
        kid = client.get_or_create_kernel()
        assert kid == "kernel-abc-123"
        assert client.kernel_id == "kernel-abc-123"

        # Subsequent call returns cached kernel_id
        assert client.get_or_create_kernel() == "kernel-abc-123"
        assert mock_get.call_count == 1


def test_get_or_create_kernel_spawn() -> None:
    """Test get_or_create_kernel when no kernel exists and one is created."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with (
        patch.object(client._session, "get") as mock_get,
        patch.object(client._session, "post") as mock_post,
    ):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [])
        mock_post.return_value = MagicMock(
            status_code=201, json=lambda: {"id": "new-kernel-999"}
        )

        kid = client.get_or_create_kernel()
        assert kid == "new-kernel-999"


def test_interrupt_kernel() -> None:
    """Test interrupt sends interrupt POST request."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    client.kernel_id = "test-kernel"
    with patch.object(client._session, "post") as mock_post:
        mock_post.return_value = MagicMock(status_code=204)
        assert client.interrupt() is True
        mock_post.assert_called_once()


def test_upload_file(tmp_path: Path) -> None:
    """Test file upload to Jupyter contents API."""
    test_file = tmp_path / "model.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "put") as mock_put:
        mock_put.return_value = MagicMock(status_code=201)
        ok = client.upload_file(test_file, "uploaded_model.py")
        assert ok is True

        mock_put.assert_called_once()
        args, kwargs = mock_put.call_args
        assert "/api/contents/uploaded_model.py" in args[0]
        payload = kwargs["json"]
        assert payload["type"] == "file"
        assert payload["format"] == "base64"
        assert base64.b64decode(payload["content"]).decode("utf-8") == "print('hello')"


def test_upload_file_not_found(tmp_path: Path) -> None:
    """Test upload_file raises FileNotFoundError for nonexistent local path."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with pytest.raises(FileNotFoundError):
        client.upload_file(tmp_path / "nonexistent.py", "remote.py")


def test_download_file(tmp_path: Path) -> None:
    """Test file download from Jupyter contents API."""
    dest_path = tmp_path / "output" / "weights.bin"

    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "get") as mock_get:
        encoded_data = base64.b64encode(b"binary_model_bytes").decode("ascii")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"format": "base64", "content": encoded_data},
        )

        result = client.download_file("weights.bin", dest_path)
        assert result == dest_path
        assert dest_path.read_bytes() == b"binary_model_bytes"


def test_list_files() -> None:
    """Test directory contents listing."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client._session, "get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [
                    {
                        "name": "train.py",
                        "path": "train.py",
                        "type": "file",
                        "size": 1024,
                        "last_modified": "2026-09-16T10:00:00Z",
                    },
                    {
                        "name": "models",
                        "path": "models",
                        "type": "directory",
                        "size": None,
                        "last_modified": "2026-09-16T10:00:00Z",
                    },
                ]
            },
        )
        entries = client.list_files("")
        assert len(entries) == 2
        assert entries[0]["name"] == "train.py"
        assert entries[1]["type"] == "directory"


def test_execute_code_success() -> None:
    """Test WebSocket code execution with streaming outputs and idle finish."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    client.kernel_id = "test-kernel-123"

    mock_ws = MagicMock()

    received_output: list[str] = []

    def mock_recv_generator():
        # First call is made inside execute after send
        # We need parent msg_id from what was sent
        sent_payload = json.loads(mock_ws.send.call_args[0][0])
        msg_id = sent_payload["header"]["msg_id"]

        yield json.dumps(
            {
                "parent_header": {"msg_id": msg_id},
                "msg_type": "stream",
                "content": {"text": "Step 1/10 completed\n"},
            }
        )
        yield json.dumps(
            {
                "parent_header": {"msg_id": msg_id},
                "msg_type": "execute_result",
                "content": {"data": {"text/plain": "42"}},
            }
        )
        yield json.dumps(
            {
                "parent_header": {"msg_id": msg_id},
                "msg_type": "status",
                "content": {"execution_state": "idle"},
            }
        )

    gen = None

    def fake_recv():
        nonlocal gen
        if gen is None:
            gen = mock_recv_generator()
        return next(gen)

    mock_ws.recv.side_effect = fake_recv

    with patch("websocket.create_connection", return_value=mock_ws):
        res = client.execute(
            "x = 42\nx", on_output=lambda chunk: received_output.append(chunk)
        )

    assert res["success"] is True
    assert "Step 1/10 completed\n" in res["output"]
    assert "42\n" in res["output"]
    assert received_output == ["Step 1/10 completed\n", "42\n"]
    mock_ws.close.assert_called_once()


def test_execute_code_error() -> None:
    """Test WebSocket execution with Python exception/traceback."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    client.kernel_id = "test-kernel-123"

    mock_ws = MagicMock()
    gen = None

    def mock_recv_generator():
        sent_payload = json.loads(mock_ws.send.call_args[0][0])
        msg_id = sent_payload["header"]["msg_id"]

        yield json.dumps(
            {
                "parent_header": {"msg_id": msg_id},
                "msg_type": "error",
                "content": {
                    "ename": "ZeroDivisionError",
                    "evalue": "division by zero",
                    "traceback": [
                        "Traceback (most recent call last):",
                        "ZeroDivisionError: division by zero",
                    ],
                },
            }
        )
        yield json.dumps(
            {
                "parent_header": {"msg_id": msg_id},
                "msg_type": "status",
                "content": {"execution_state": "idle"},
            }
        )

    def fake_recv():
        nonlocal gen
        if gen is None:
            gen = mock_recv_generator()
        return next(gen)

    mock_ws.recv.side_effect = fake_recv

    with patch("websocket.create_connection", return_value=mock_ws):
        res = client.execute("1 / 0")

    assert res["success"] is False
    assert res["error_name"] == "ZeroDivisionError"
    assert "ZeroDivisionError: division by zero" in res["output"]


def test_get_gpu_info() -> None:
    """Test get_gpu_info queries PyTorch CUDA diagnostics."""
    client = JupyterProxyClient("https://proxy.test/proxy")
    with patch.object(client, "execute") as mock_exec:
        mock_exec.return_value = {
            "success": True,
            "output": "CUDA Available: True\nGPU Count: 2\n",
        }
        info = client.get_gpu_info()
        assert "CUDA Available: True" in info
        mock_exec.assert_called_once()
