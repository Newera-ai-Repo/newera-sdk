"""The Newera client: one object for the API key and the URL.

    from newera import Newera

    client = Newera()                       # NEWERA_API_KEY from the environment
    async with client.avatar.connect() as call:
        ...
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

from .avatar import DEFAULT_BASE_URL, Avatar, error_for


class Newera:
    """`api_key` defaults to the NEWERA_API_KEY environment variable. `base_url` is the HTTPS address of the
    API; the WebSocket address is derived from it."""

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, headers: dict | None = None):
        self.api_key = api_key or os.environ.get("NEWERA_API_KEY")
        self.base_url = (base_url or os.environ.get("NEWERA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.headers = dict(headers or {})
        self.avatar = Avatar(self)

    @property
    def avatar_ws_url(self) -> str:
        base = self.base_url
        for http, ws in (("https://", "wss://"), ("http://", "ws://")):
            if base.startswith(http):
                base = ws + base[len(http):]
        return base + "/v1/avatar"

    def _ssl_context(self) -> ssl.SSLContext | None:
        """TLS for https/wss, with the CA bundle of `certifi`: a Python from python.org on macOS ships
        without root certificates and would fail with CERTIFICATE_VERIFY_FAILED."""
        if not self.base_url.startswith("https://"):
            return None
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:  # pragma: no cover
            return ssl.create_default_context()

    def _headers(self) -> dict:
        h = dict(self.headers)
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def _http(self, method: str, path: str, body: dict | None = None) -> dict:
        """A blocking JSON request (run it in a thread). HTTP errors raise the matching AvatarError."""
        if not self.api_key:
            raise error_for("invalid_api_key", "No API key. Pass Newera(api_key=...) or set NEWERA_API_KEY.", http_status=401)
        headers = self._headers()
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20, context=self._ssl_context()) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode()).get("error") or {}
            except Exception:  # noqa: BLE001 - not JSON
                err = {}
            raise error_for(err.get("code"), err.get("message") or "HTTP %s" % exc.code, http_status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise error_for("connection_error", "Could not reach %s: %s" % (self.base_url, exc.reason)) from exc
