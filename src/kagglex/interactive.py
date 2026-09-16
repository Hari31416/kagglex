"""Interactive Jupyter Proxy client for sub-second REPL execution and file operations."""

import base64
import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import websocket

logger = logging.getLogger(__name__)


class JupyterProxyClient:
    """Connects to a running Kaggle Jupyter Server proxy via REST and WebSocket."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 120,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self.default_timeout = timeout
        self.on_output = on_output
        self.session_id = uuid.uuid4().hex
        self.kernel_id: str | None = None
        self._session = requests.Session()

        parsed = urlparse(base_url.strip())
        self._token: str | None = None

        if parsed.query and "token=" in parsed.query:
            query_params = parse_qs(parsed.query)
            self._token = query_params.get("token", [None])[0]
            self.base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip(
                "/"
            )
            self._url_format = "query"
        else:
            clean_url = base_url.strip().rstrip("/")
            if not clean_url.endswith("/proxy"):
                clean_url = clean_url + "/proxy"
            self.base_url = clean_url
            self._url_format = "path"

        websocket.enableTrace(False)

    def _build_url(self, path: str) -> str:
        """Construct full REST URL with authentication."""
        path = "/" + path.lstrip("/")
        if self._url_format == "query" and self._token:
            sep = "&" if "?" in path else "?"
            return f"{self.base_url}{path}{sep}token={self._token}"
        return f"{self.base_url}{path}"

    def _build_ws_url(self, path: str) -> str:
        """Construct WebSocket URL with authentication."""
        path = "/" + path.lstrip("/")
        ws_base = self.base_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        if self._url_format == "query" and self._token:
            sep = "&" if "?" in path else "?"
            return f"{ws_base}{path}{sep}token={self._token}"
        return f"{ws_base}{path}"

    def test_connection(self) -> bool:
        """Verify connectivity to Kaggle Jupyter API."""
        try:
            r = self._session.get(self._build_url("/api"), timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.debug("Jupyter connection check failed: %s", e)
            return False

    def get_or_create_kernel(self) -> str:
        """Discover existing running kernel ID or launch a new Python kernel."""
        if self.kernel_id:
            return self.kernel_id

        # Check existing kernels
        try:
            r = self._session.get(self._build_url("/api/kernels"), timeout=10)
            if r.status_code == 200:
                kernels = r.json()
                if kernels and isinstance(kernels, list):
                    self.kernel_id = kernels[0].get("id")
                    logger.debug("Found active kernel: %s", self.kernel_id)
                    return str(self.kernel_id)
        except Exception as e:
            logger.debug("Error querying running kernels: %s", e)

        # Create new kernel if none found
        try:
            r = self._session.post(
                self._build_url("/api/kernels"),
                json={"name": "python3"},
                timeout=15,
            )
            if r.status_code in (200, 201):
                data = r.json()
                self.kernel_id = data.get("id")
                logger.debug("Spawned new kernel: %s", self.kernel_id)
                return str(self.kernel_id)
        except Exception as e:
            logger.error("Failed to start kernel on Jupyter server: %s", e)
            raise

        raise RuntimeError("Unable to acquire or start a Jupyter kernel.")

    def interrupt(self) -> bool:
        """Interrupt a currently executing kernel."""
        kid = self.kernel_id or self.get_or_create_kernel()
        try:
            r = self._session.post(
                self._build_url(f"/api/kernels/{kid}/interrupt"), timeout=10
            )
            return r.status_code in (200, 204)
        except Exception as e:
            logger.warning("Could not interrupt kernel %s: %s", kid, e)
            return False

    def execute(
        self,
        code: str,
        timeout: int | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Execute Python code on the remote Jupyter kernel via WebSocket.

        Args:
            code: Python code string to execute.
            timeout: Maximum seconds to wait for execution to complete.
            on_output: Optional streaming callback for stdout/stderr chunks.

        Returns:
            Dict containing output text, success status, and error details.
        """
        kid = self.get_or_create_kernel()
        ws_url = self._build_ws_url(f"/api/kernels/{kid}/channels")
        effective_timeout = timeout or self.default_timeout

        ws = websocket.create_connection(ws_url, timeout=15)
        msg_id = uuid.uuid4().hex

        execute_msg = {
            "header": {
                "msg_id": msg_id,
                "username": "kagglex",
                "session": self.session_id,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
        }

        ws.send(json.dumps(execute_msg))

        output_chunks: list[str] = []
        has_error = False
        error_name: str | None = None
        error_value: str | None = None
        traceback_list: list[str] = []

        start_time = time.time()

        try:
            while True:
                if time.time() - start_time > effective_timeout:
                    raise TimeoutError(
                        f"Execution timed out after {effective_timeout}s."
                    )

                ws.settimeout(5.0)
                try:
                    raw_msg = ws.recv()
                except (websocket.WebSocketTimeoutException, TimeoutError):
                    continue

                if not raw_msg:
                    continue

                msg = json.loads(raw_msg)
                parent_id = (
                    msg.get("parent_header", {}).get("msg_id")
                    if msg.get("parent_header")
                    else None
                )
                if parent_id != msg_id:
                    continue

                msg_type = msg.get("msg_type")
                content = msg.get("content", {})

                if msg_type == "stream":
                    text = content.get("text", "")
                    output_chunks.append(text)
                    if on_output:
                        on_output(text)
                    elif self.on_output:
                        self.on_output(text)

                elif msg_type in ("execute_result", "display_data"):
                    data = content.get("data", {})
                    text_plain = data.get("text/plain", "")
                    if text_plain:
                        text = text_plain + "\n"
                        output_chunks.append(text)
                        if on_output:
                            on_output(text)
                        elif self.on_output:
                            self.on_output(text)

                elif msg_type == "error":
                    has_error = True
                    error_name = content.get("ename")
                    error_value = content.get("evalue")
                    traceback_list = content.get("traceback", [])
                    err_text = "\n".join(traceback_list) + "\n"
                    output_chunks.append(err_text)
                    if on_output:
                        on_output(err_text)
                    elif self.on_output:
                        self.on_output(err_text)

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break
        finally:
            ws.close()

        full_output = "".join(output_chunks)
        return {
            "success": not has_error,
            "output": full_output,
            "error_name": error_name,
            "error_value": error_value,
            "traceback": traceback_list,
        }

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload a local file to remote Jupyter environment.

        Args:
            local_path: Local file path.
            remote_path: Target path relative to /kaggle/working/.
        """
        local_p = local_path.resolve()
        if not local_p.is_file():
            raise FileNotFoundError(f"Local file not found: {local_p}")

        content_b64 = base64.b64encode(local_p.read_bytes()).decode("ascii")
        clean_remote = remote_path.lstrip("/")

        payload = {
            "type": "file",
            "format": "base64",
            "content": content_b64,
        }
        r = self._session.put(
            self._build_url(f"/api/contents/{clean_remote}"),
            json=payload,
            timeout=30,
        )
        return r.status_code in (200, 201)

    def download_file(self, remote_path: str, local_path: Path) -> Path:
        """Download a file from remote Jupyter environment.

        Args:
            remote_path: Remote path relative to /kaggle/working/.
            local_path: Local target file path.
        """
        clean_remote = remote_path.lstrip("/")
        r = self._session.get(
            self._build_url(f"/api/contents/{clean_remote}"), timeout=30
        )
        if r.status_code != 200:
            raise FileNotFoundError(
                f"Remote file '{remote_path}' not found (HTTP {r.status_code})"
            )

        data = r.json()
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if data.get("format") == "base64":
            local_path.write_bytes(base64.b64decode(data.get("content", "")))
        else:
            local_path.write_text(data.get("content", ""), encoding="utf-8")

        return local_path

    def list_files(self, remote_path: str = "") -> list[dict[str, Any]]:
        """List files and folders in the remote working directory."""
        clean_path = remote_path.lstrip("/")
        r = self._session.get(
            self._build_url(f"/api/contents/{clean_path}"), timeout=15
        )
        if r.status_code != 200:
            return []

        data = r.json()
        entries = data.get("content", [])
        results = []
        if isinstance(entries, list):
            for item in entries:
                results.append(
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size": item.get("size"),
                        "last_modified": item.get("last_modified"),
                    }
                )
        return results

    def get_gpu_info(self) -> str:
        """Query GPU and PyTorch diagnostics on the active Jupyter kernel."""
        code = (
            "import torch\n"
            "print('=' * 50)\n"
            "print(f'PyTorch Version: {torch.__version__}')\n"
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n"
            "if torch.cuda.is_available():\n"
            "    count = torch.cuda.device_count()\n"
            "    print(f'GPU Count: {count}')\n"
            "    for i in range(count):\n"
            "        p = torch.cuda.get_device_properties(i)\n"
            "        print(f'  GPU [{i}]: {p.name} ({p.total_memory / (1024**3):.2f} GB)')\n"
            "else:\n"
            "    print('No CUDA GPU available.')\n"
            "print('=' * 50)\n"
        )
        res = self.execute(code)
        return str(res.get("output", ""))
