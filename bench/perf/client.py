"""Client HTTP d'inférence : chat (avec streaming), télémétrie serveur, sondes.

Cible tout serveur **OpenAI-compatible** exposant `/v1/chat/completions` : `llama-server`
(llama.cpp), Strata (`serve/server.py`, mêmes `timings` que llama.cpp), `llama-cpp-python`, vLLM…
Aucune supposition n'est faite sur les champs optionnels : tout est lu défensivement, et ce qui
manque est **absent** du résultat plutôt qu'inventé.

Ce que ce module cherche à rendre impossible
--------------------------------------------
Deux pièges rendent une mesure de prefill fausse sans qu'on s'en aperçoive :

1. **le cache de prompt** — si la requête est identique à une précédente, le serveur réutilise le
   préfixe et le prefill mesuré s'effondre. `unique_nonce` ajoute un marqueur unique par requête, et
   `timings.cache_n` est relevé pour pouvoir le **détecter** ; l'un ne remplace pas l'autre.
2. **la mesure sans streaming** — sans streaming il n'existe aucun instant « premier token » : le
   seul temps disponible est la durée totale. `stream=True` est donc le défaut, et la latence au
   premier token est chronométrée côté client, en plus du `prompt_ms` du serveur.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

DEFAULT_TIMEOUT_S = 3600.0

#: Champs de `timings` (llama.cpp et Strata) que l'on conserve s'ils sont présents.
TIMING_KEYS = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
               "predicted_per_second", "cache_n", "draft_n", "draft_n_accepted")


@dataclass
class Response:
    """Résultat d'une requête de génération. Chaque champ absent du serveur reste à `None`."""

    content: str = ""
    reasoning: str = ""                  # canal de raisonnement (`delta.reasoning_content`)
    timings: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    ttft_s: float | None = None          # premier token, quel que soit le canal
    ttft_answer_s: float | None = None   # premier token de **réponse** (hors raisonnement)
    total_s: float = 0.0
    chunks: int = 0
    ok: bool = True
    error: str | None = None
    http_status: int | None = None

    @property
    def prompt_tokens(self) -> int | None:
        return self.usage.get("prompt_tokens") if isinstance(self.usage.get("prompt_tokens"), int) else None

    @property
    def completion_tokens(self) -> int | None:
        return self.usage.get("completion_tokens") if isinstance(self.usage.get("completion_tokens"), int) else None

    def rate(self, key: str) -> float | None:
        """Débit serveur : `prompt_per_second` (prefill) ou `predicted_per_second` (décodage)."""
        value = self.timings.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "error": self.error, "http_status": self.http_status,
                "ttft_s": self.ttft_s, "ttft_answer_s": self.ttft_answer_s,
                "total_s": round(self.total_s, 4), "chunks": self.chunks,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "timings": self.timings, "usage": self.usage,
                "content_chars": len(self.content), "reasoning_chars": len(self.reasoning)}


class ChatClient:
    """Client minimal, sans dépendance, pour un serveur d'inférence OpenAI-compatible."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._tok_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ bas niveau

    def _open(self, path: str, payload: dict | None = None, method: str = "POST",
              timeout: float | None = None):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data, {"Content-Type": "application/json",
                                                "Accept": "text/event-stream, application/json"})
        req.get_method = lambda: method
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def _json(self, path: str, payload: dict | None = None, method: str = "POST",
              timeout: float = 30.0) -> dict | None:
        """Requête JSON courte. Retourne `None` si l'endpoint n'existe pas ou répond mal.

        Sonder n'est pas mesurer : une sonde indisponible doit dégrader proprement, jamais
        interrompre une campagne.
        """
        try:
            with self._open(path, payload, method, timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, OSError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    # ------------------------------------------------------------------ sondes

    def health(self) -> bool:
        try:
            with self._open("/health", None, "GET", timeout=5.0) as resp:
                return resp.status == 200
        except (urllib.error.HTTPError, OSError):
            return False

    def props(self) -> dict | None:
        """`/props` : build, modèle, `n_ctx`, réglages de génération. Absent chez certains moteurs."""
        return self._json("/props", None, "GET")

    def slots(self) -> list | None:
        body = self._json("/slots", None, "GET")
        return body if isinstance(body, list) else None

    def metrics(self) -> str | None:
        """`/metrics` : Prometheus si disponible (Strata expose `strata:live_tok_s`, etc.)."""
        try:
            req = urllib.request.Request(self.base_url + "/metrics",
                                         headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return resp.read().decode("utf-8", "replace")
        except (urllib.error.HTTPError, OSError):
            return None

    def tokenize(self, text: str) -> int | None:
        """Nombre de tokens selon le **vrai** tokenizer du serveur (`POST /tokenize`), avec cache."""
        key = str(hash(text))
        if key in self._tok_cache:
            return self._tok_cache[key]
        body = self._json("/tokenize", {"content": text, "add_special": False})
        tokens = body.get("tokens") if body else None
        if not isinstance(tokens, list):
            return None
        self._tok_cache[key] = len(tokens)
        return len(tokens)

    # ------------------------------------------------------------------ génération

    def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.0, seed: int = 0,
             stream: bool = True, unique_nonce: bool = True, extra: dict | None = None,
             timeout: float | None = None) -> Response:
        """Une génération. `timings` et `usage` sont relevés quel que soit le mode.

        `unique_nonce=True` ajoute un marqueur unique en fin de dernier message : sans lui, le
        serveur peut réutiliser le préfixe en cache et le prefill mesuré n'est plus un prefill.
        """
        msgs = [dict(m) for m in messages]
        nonce = None
        if unique_nonce and msgs:
            nonce = f"\n\n[réf. mesure {uuid.uuid4().hex[:12]}]"
            msgs[-1]["content"] = str(msgs[-1].get("content", "")) + nonce
        payload = {"model": "mesure", "messages": msgs, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed, "stream": stream}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if extra:
            payload.update(extra)

        started = time.perf_counter()
        if not stream:
            try:
                with self._open("/v1/chat/completions", payload, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                return Response(ok=False, error=f"HTTP {exc.code}", http_status=exc.code,
                                total_s=time.perf_counter() - started)
            except OSError as exc:
                return Response(ok=False, error=f"serveur injoignable ({exc})",
                                total_s=time.perf_counter() - started)
            return Response(content=body["choices"][0]["message"].get("content") or "",
                            reasoning=body["choices"][0]["message"].get("reasoning_content") or "",
                            timings=_timings(body.get("timings")), usage=body.get("usage") or {},
                            total_s=time.perf_counter() - started,
                            http_status=getattr(resp, "status", None))

        return self._stream(payload, started, timeout, nonce)

    def _stream(self, payload: dict, started: float, timeout: float | None,
                nonce: str | None) -> Response:
        """Lecture SSE.

        Deux précautions apprises sur un vrai serveur (Strata) :

        - **le canal de raisonnement compte.** Un modèle qui « réfléchit » émet
          `delta.reasoning_content` avant tout `delta.content`. Ne lire que `content` faisait
          mesurer le débit du raisonnement en croyant mesurer celui de la réponse, et laissait
          `ttft` à `None` alors que le flux était parfaitement lisible. Les deux canaux sont donc
          suivis séparément, avec une latence au premier token **de chaque canal**.
        - **un flux vide n'est pas un succès.** Parse des chunks sans jamais obtenir de texte doit
          produire une erreur explicite, pas une réponse « ok » et muette.
        """
        out = Response()
        try:
            resp = self._open("/v1/chat/completions", payload, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except OSError:
                pass
            return Response(ok=False, error=f"HTTP {exc.code} {detail}".strip(),
                            http_status=exc.code, total_s=time.perf_counter() - started)
        except OSError as exc:
            return Response(ok=False, error=f"serveur injoignable ({exc})",
                            total_s=time.perf_counter() - started)
        with resp:
            out.http_status = getattr(resp, "status", None)
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):     # `: keep-alive` et commentaires SSE ignorés
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                out.chunks += 1
                if isinstance(event.get("timings"), dict):
                    out.timings.update(_timings(event["timings"]))
                if isinstance(event.get("usage"), dict) and event["usage"]:
                    out.usage.update(event["usage"])
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    thought, piece = delta.get("reasoning_content"), delta.get("content")
                    if thought:
                        if out.ttft_s is None:
                            out.ttft_s = time.perf_counter() - started
                        out.reasoning += thought
                    if piece:
                        if out.ttft_s is None:
                            out.ttft_s = time.perf_counter() - started
                        if out.ttft_answer_s is None:
                            out.ttft_answer_s = time.perf_counter() - started
                        out.content += piece
        out.total_s = time.perf_counter() - started
        if nonce:
            out.content = out.content.replace(nonce.strip(), "")
        if not out.content and not out.reasoning:
            out.ok, out.error = False, ("flux sans texte : aucun contenu ni raisonnement reçu "
                                        f"({out.chunks} chunk(s) analysé(s))")
        return out


def _timings(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {k: raw[k] for k in TIMING_KEYS if isinstance(raw.get(k), (int, float))}
