"""Adaptateurs de modèles RÉELS : Ollama (HTTP local) et manuel (copier-coller, ex. DeepSeek web).

Même interface que les adaptateurs factices (declare / plan / step / finalize), mais :
  - NON déterministes par construction : la reprise (étape n depuis le seul compte rendu) est une
    MESURE de reproductibilité du modèle, pas une garantie du code ;
  - le texte du contexte chargé est transmis via info["context_text"] ;
  - Ollama : appel à l'intérieur de EXECUTER seulement, température 0 et graine fixe, `num_ctx` = limite
    déclarée, `num_predict` = budget, réflexion désactivée ; chaque réponse est mise en cache sous
    info["cache_dir"] (rejouer une exécution interrompue ne repaie pas les étapes faites) ;
  - llama.cpp : lance `llama-server` (un GGUF, `-c` = limite déclarée, `-ngl` réglable) sur 127.0.0.1 au premier
    appel non caché, l'arrête à `close()` (le harnais le ferme après chaque modèle : la VRAM est libérée avant
    le suivant) ; ou cible un serveur déjà lancé (`params.host`, jamais arrêté par le harnais) ;
  - manuel : écrit step<k>.prompt.txt, lit step<k>.response.txt dans info["manual_dir"]/<modèle>/ ;
    s'arrête avec un message clair tant que la réponse manque (relancer avec --resume).
Les tokens consommés restent mesurés par l'estimateur du harnais (chars/4) ; les compteurs réels du
serveur sont consignés dans `usage` (journal `live_usage`) pour calibrer cet estimateur.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .budget import STEP_NAMES
from .trace import canonical_json

OLLAMA_URL = "http://localhost:11434"
TIMEOUT_S = 1800  # modèles 27-35B partiellement sur CPU : lents

STEP_SPECS = {
    1: ("meta, executive_verdict, formalization",
        "meta = {\"model\": <nom du modèle>} ; executive_verdict = {\"decision\": \"GO\" ou \"NO-GO\", \"summary\": texte} ; "
        "formalization = objet non vide (G, nœuds, arêtes, relations, S_t, Advance, Merge, Compress, ProjectToLLM)."),
    2: ("correlation_map, anti_mystical_audit, fruitful_intuitions",
        "correlation_map = {\"solid\": [...], \"fragile\": [...], \"illusory\": [...]} classant les axes "
        "\"word~etymon\", \"etymon~phoneme\", \"word~phoneme\" ; anti_mystical_audit = {\"sdm\": nombre 0-100, "
        "\"justification\": texte} ; fruitful_intuitions = 3 à 7 éléments {\"text\", \"sif\": nombre 0-100, \"justification\"}."),
    3: ("experimental_plan, system_integration, smallest_decisive_experiment, claim_tagging_summary",
        "experimental_plan = {\"baselines\": [les 9 baselines], \"controls\": [les 4 contrôles]} ; system_integration "
        "= objet non vide (points d'intégration llama.cpp, coûts cachés) ; smallest_decisive_experiment = objet non vide "
        "avec un test de réfutation ; claim_tagging_summary = {\"claims\": [{\"text\", \"tag\" parmi DEFINI, TESTABLE, "
        "PLAUSIBLE, SPECULATIF, PROBABLEMENT_FAUX, NON_FALSIFIABLE, et \"refutation_test\" ou \"justification\"}]} "
        "dont au moins une claim avec \"refutation_test\"."),
}


def step_prompt(k: int, prior: dict, info: dict) -> str:
    keys, convention = STEP_SPECS[k]
    done = f"\n\nÉtapes déjà produites (à respecter, ne pas répéter) :\n{canonical_json(prior)}" if prior else ""
    return (f"Étape {k}/{len(STEP_NAMES)} ({STEP_NAMES[k - 1]}). Tu t'appelles « {info['model']} ».\n"
            f"Réponds par UN SEUL objet JSON, sans aucun texte autour, contenant exactement les clés : {keys}.\n"
            f"Convention du harnais : {convention}\nLimite de réponse : {info.get('budget', '?')} tokens.{done}")


def parse_step(text: str) -> tuple[dict, bool]:
    """(sortie de l'étape, balise ``` retirée). Texte non JSON : marqueur `_texte_hors_json` (rejet à VALIDER)."""
    raw, fenced = text.strip(), False
    if raw.startswith("```") and raw.endswith("```"):
        raw, fenced = raw.strip("`").removeprefix("json").strip(), True
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"_texte_hors_json": text}, False
    return (obj, fenced) if isinstance(obj, dict) else ({"_texte_hors_json": text}, False)


class LiveAdapter:
    kind = "live"
    deterministic = False
    #: Un modèle réel **n'annonce pas** de budget : `plan()` renvoie une répartition neutre
    #: (`budget / 3`) décidée par le harnais. L'écart à ce chiffre ne mesure donc pas l'honnêteté du
    #: modèle et ne doit pas servir à l'exclure du consensus (il reste publié comme métrique).
    declares_budget = False

    def __init__(self, context_limit: int, **params) -> None:
        self.context_limit = context_limit
        self.params = params
        self.usage: list[dict] = []

    def declare(self) -> dict:
        return {"context_limit": self.context_limit}

    def count_tokens(self, text: str, info: dict) -> tuple[int, str] | None:
        """Comptage exact du texte par le serveur, avant exécution.

        Retourne `(tokens, source)` ou `None` si le moteur n'expose pas de tokenizer. Sert à
        vérifier la garde de contexte avec le **vrai** tokenizer au lieu de `chars/4`, qui
        sous-compte (mesuré : ~10 % sur le contexte, jusqu'à 35 % sur le prompt complet).
        """
        return None

    def plan(self, info: dict) -> list[int]:
        """Estimation neutre : le budget réparti également sur les étapes."""
        return [max(1, info["budget"] // len(STEP_NAMES))] * len(STEP_NAMES)

    def finalize(self, merged: dict) -> str:
        if "_texte_hors_json" in merged:
            return merged["_texte_hors_json"]
        return canonical_json(merged)

    def _complete(self, k: int, prior: dict, info: dict) -> tuple[str, dict]:
        raise NotImplementedError

    def close(self) -> None:
        """Libère les ressources (serveur lancé par l'adaptateur)."""

    @staticmethod
    def _cached(payload: dict, info: dict, fetch) -> tuple[str, dict]:
        """Rejoue la réponse mise en cache pour cette requête exacte, sinon appelle `fetch()` -> (texte, usage)."""
        cache = None
        if info.get("cache_dir"):
            key = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            cache = Path(info["cache_dir"]) / f"{key}.json"
            if cache.exists():
                hit = json.loads(cache.read_text(encoding="utf-8"))
                return hit["content"], {**hit["usage"], "cached": True}
        content, usage = fetch()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"content": content, "usage": usage}, ensure_ascii=False),
                             encoding="utf-8")
        return content, {**usage, "cached": False}

    @staticmethod
    def _post(url: str, payload: dict, label: str) -> dict:
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"{label} : {url} a répondu {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}")
        except OSError as exc:
            raise SystemExit(f"{label} : serveur injoignable sur {url} ({exc})")

    @staticmethod
    def _post_safe(url: str, payload: dict) -> dict | None:
        """Comme `_post`, mais retourne `None` au lieu d'interrompre le harnais.

        Réservé aux **sondes facultatives** (comptage de tokens) : leur indisponibilité doit
        dégrader proprement vers l'estimateur, jamais tuer une campagne.
        """
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, OSError, ValueError):
            return None
        return body if isinstance(body, dict) else None

    def step(self, k: int, prior: dict, info: dict) -> dict:
        text, usage = self._complete(k, prior, info)
        out, fenced = parse_step(text)
        if k == 1 and isinstance(out.get("meta"), dict):
            out["meta"].setdefault("model", info["model"])
        self.usage.append({"step": k, **usage, "fence_stripped": fenced})
        return out


class OllamaAdapter(LiveAdapter):
    kind = "ollama"

    def _complete(self, k: int, prior: dict, info: dict) -> tuple[str, dict]:
        model = self.params["model"]
        payload = {
            "model": model, "stream": False, "think": False, "format": "json", "keep_alive": "1m",
            "options": {"num_ctx": self.context_limit, "temperature": 0, "seed": info["seed"],
                        "num_predict": info["budget"]},
            "messages": [{"role": "system", "content": info["context_text"]},
                         {"role": "user", "content": step_prompt(k, prior, info)}],
        }
        url = self.params.get("host", OLLAMA_URL).rstrip("/") + "/api/chat"

        def fetch():
            body = self._post(url, payload, model)
            evals, dur = body.get("eval_count", 0), body.get("eval_duration", 0)
            usage = {"prompt_tokens_real": body.get("prompt_eval_count"), "completion_tokens_real": evals,
                     "tokens_per_s": round(evals / (dur / 1e9), 2) if dur else None,
                     "done_reason": body.get("done_reason")}
            return body["message"]["content"], usage

        return self._cached(payload, info, fetch)


class LlamaCppAdapter(LiveAdapter):
    """llama-server (llama.cpp) : API OpenAI-compatible `/v1/chat/completions`.

    params : `gguf` (chemin, requis sauf avec `host`), `ngl` (couches en VRAM, défaut 99), `host` (serveur déjà
    lancé : jamais arrêté), `server` (exécutable, défaut : llama-server du PATH),
    `extra_args` (liste d'options supplémentaires, ex. ["--n-cpu-moe", "20", "-ctk", "q8_0"]).
    """
    kind = "llamacpp"
    START_TIMEOUT_S = 900  # chargement d'un GGUF de 15-20 Go

    def __init__(self, context_limit: int, **params) -> None:
        super().__init__(context_limit, **params)
        self._proc: subprocess.Popen | None = None
        self._base: str | None = params.get("host")
        self._log = None
        self._tok_cache: dict[str, tuple[int, str]] = {}

    def count_tokens(self, text: str, info: dict) -> tuple[int, str] | None:
        """Comptage réel via `POST /tokenize` de `llama-server` (cache par hash de texte).

        Retourne `None` si l'endpoint est absent (par ex. un moteur qui n'expose pas de tokenizer) :
        le harnais retombe alors sur `chars/4` et le consigne.
        """
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._tok_cache:
            return self._tok_cache[key]
        body = self._post_safe(self._start(info) + "/tokenize",
                               {"content": text, "add_special": False})
        if body is None or not isinstance(body.get("tokens"), list):
            return None
        result = (len(body["tokens"]), "server /tokenize")
        self._tok_cache[key] = result
        return result

    def command(self, port: int) -> list[str]:
        exe = self.params.get("server") or shutil.which("llama-server")
        if not exe:
            raise SystemExit("llama-server introuvable dans le PATH (params.server pour le chemin explicite)")
        return [exe, "-m", self.params["gguf"], "-c", str(self.context_limit),
                "-ngl", str(self.params.get("ngl", 99)), "-np", "1", "--host", "127.0.0.1", "--port", str(port),
                "--jinja", "--reasoning-budget", "0", *map(str, self.params.get("extra_args", []))]

    def _start(self, info: dict) -> str:
        if self._base:
            return self._base.rstrip("/")
        if self._proc is None:
            gguf = Path(self.params["gguf"])
            if not gguf.is_file():
                raise SystemExit(f"GGUF introuvable : {gguf}")
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            log_path = Path(info.get("out_dir") or ".") / f"llama-server-{info['model']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open("ab")
            self._proc = subprocess.Popen(self.command(port), stdout=self._log, stderr=subprocess.STDOUT)
            atexit.register(self.close)
            self._base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + self.START_TIMEOUT_S
            while True:
                if self._proc.poll() is not None:
                    raise SystemExit(f"llama-server s'est arrêté (code {self._proc.returncode}), voir {log_path}")
                try:
                    with urllib.request.urlopen(self._base + "/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    self.close()
                    raise SystemExit(f"llama-server : démarrage > {self.START_TIMEOUT_S}s, voir {log_path}")
                time.sleep(2)
        return self._base

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc, self._base = None, self.params.get("host")
        if self._log is not None:
            self._log.close()
            self._log = None

    def _complete(self, k: int, prior: dict, info: dict) -> tuple[str, dict]:
        name = Path(self.params.get("gguf", "serveur")).name
        payload = {
            "model": name, "stream": False, "temperature": 0, "seed": info["seed"], "max_tokens": info["budget"],
            "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False},
            "n_ctx": self.context_limit, "ngl": self.params.get("ngl", 99),  # dans la clé de cache seulement
            "messages": [{"role": "system", "content": info["context_text"]},
                         {"role": "user", "content": step_prompt(k, prior, info)}],
        }

        def fetch():
            body = self._post(self._start(info) + "/v1/chat/completions",
                              {k_: v for k_, v in payload.items() if k_ not in ("n_ctx", "ngl")}, name)
            use, timings = body.get("usage", {}), body.get("timings", {})
            usage = {"prompt_tokens_real": use.get("prompt_tokens"),
                     "completion_tokens_real": use.get("completion_tokens"),
                     "tokens_per_s": round(timings["predicted_per_second"], 2) if "predicted_per_second" in timings else None,
                     "done_reason": body["choices"][0].get("finish_reason")}
            return body["choices"][0]["message"]["content"], usage

        return self._cached(payload, info, fetch)


class ManualAdapter(LiveAdapter):
    kind = "manuel"

    def _complete(self, k: int, prior: dict, info: dict) -> tuple[str, dict]:
        folder = Path(info["manual_dir"]) / info["model"]
        folder.mkdir(parents=True, exist_ok=True)
        prompt = folder / f"step{k}.prompt.txt"
        response = folder / f"step{k}.response.txt"
        prompt.write_text(info["context_text"] + "\n\n---\n\n" + step_prompt(k, prior, info), encoding="utf-8")
        if not response.exists():
            raise SystemExit(f"{info['model']} : colle le prompt de {prompt} dans le modèle, enregistre sa réponse "
                             f"dans {response}, puis relance la même commande avec --resume.")
        return response.read_text(encoding="utf-8"), {"manual": True}
