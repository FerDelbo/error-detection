"""Cliente do Jev (jevai.net / TypeSafe AI): ÚNICO arquivo que depende do formato da API.

ATENÇÃO: o formato abaixo segue o que o site mostra (artigo da versão 1.14: corpo JSON com
"model", "state" e "questions", cada pergunta com "type" e instruções). A documentação oficial
(docs.typesafe.ai) ainda não foi conferida. Ao ter acesso, ajuste SÓ build_payload() e
parse_response(); o resto do pipeline usa apenas JevClient.ask().

Variáveis de ambiente:
  JEV_BACKEND   api (padrão) | mock  (mock = respostas falsas e determinísticas, só para testar o encanamento)
  JEV_URL       URL completa do endpoint (obrigatória no backend api)
  JEV_API_KEY   chave (enviada como Authorization: Bearer)
  JEV_MODEL     padrão jev-1.14.0 (versão fixa; jev-latest muda sozinho)
  JEV_TIMEOUT   segundos por requisição (padrão 120)
  JEV_RETRIES   tentativas por requisição (padrão 6, espera 2^n s)
  JEV_PRICE_PER_M  US$ por milhão de tokens de entrada (padrão 0.084; a saída é grátis)

Cada pergunta é uma escolha entre CHOICES; ask() devolve, por pergunta, {"choice", "confidence"}.
Respostas ficam em cache (hash do corpo -> resposta): reexecuções não pagam de novo.
"""
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "jev-1.14.0"


class JevError(RuntimeError):
    pass


# ------------------------------------------------------------------ formato da API (ajustar com a documentação)
def build_payload(model, state, questions):
    """questions: lista de {"id", "instructions", "choices"}."""
    return {
        "model": model,
        "state": state,
        "questions": [{"id": q["id"], "type": "choice", "instructions": q["instructions"],
                       "choices": q["choices"]} for q in questions],
    }


def parse_response(body, questions):
    """Converte a resposta da API em [{"choice": str, "confidence": float}] na ordem de questions.
    Aceita algumas formas plausíveis; qualquer outra levanta JevError mostrando o corpo recebido."""
    items = None
    for key in ("answers", "results", "questions", "decisions", "outputs"):
        if isinstance(body.get(key), list):
            items = body[key]
            break
    if items is None or len(items) != len(questions):
        raise JevError(f"resposta do Jev em formato inesperado (ajuste parse_response): {json.dumps(body)[:800]}")
    out = []
    for q, it in zip(questions, items):
        choice = next((it.get(k) for k in ("choice", "answer", "value", "route", "label") if it.get(k) is not None), None)
        conf = next((it.get(k) for k in ("confidence", "score", "probability") if it.get(k) is not None), None)
        if choice is None or conf is None:
            raise JevError(f"item sem escolha/confiança (ajuste parse_response): {json.dumps(it)[:400]}")
        out.append({"choice": str(choice).strip().lower(), "confidence": float(conf)})
    return out


def input_tokens_of(body, payload):
    """Tokens de entrada informados pela API; se ausentes, estimativa de ~4 caracteres por token."""
    usage = body.get("usage") or {}
    for k in ("input_tokens", "prompt_tokens", "tokens"):
        if isinstance(usage.get(k), (int, float)):
            return int(usage[k]), False
    return len(json.dumps(payload)) // 4, True


# ------------------------------------------------------------------ cliente
class JevClient:
    def __init__(self, cache_path, usage_path):
        self.backend = os.environ.get("JEV_BACKEND", "api")
        if self.backend not in ("api", "mock"):
            raise JevError(f"JEV_BACKEND inválido: {self.backend}")
        self.url = os.environ.get("JEV_URL", "")
        self.key = os.environ.get("JEV_API_KEY", "")
        self.model = os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        self.timeout = float(os.environ.get("JEV_TIMEOUT", "120"))
        self.retries = int(os.environ.get("JEV_RETRIES", "6"))
        if self.backend == "api" and not self.url:
            raise JevError("defina JEV_URL (endpoint do Jev) ou use JEV_BACKEND=mock para um teste sem API")
        self.cache_path, self.usage_path = cache_path, usage_path
        self.lock = threading.Lock()
        self.cache = {}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec
                    except (ValueError, KeyError):
                        continue  # linha truncada por uma interrupção

    def _log(self, rec):
        with self.lock, open(self.usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _post(self, payload):
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if 400 <= e.code < 500 and e.code != 429:
                    break  # erro do pedido: repetir não resolve
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                last = f"{e.__class__.__name__}: {e}"
            time.sleep(2 ** attempt)
        raise JevError(f"Jev falhou após {self.retries} tentativas: {last}")

    @staticmethod
    def _mock(questions):
        """Respostas falsas e determinísticas (≈10% 'error'). Só para testar o pipeline sem a API."""
        out = []
        for q in questions:
            rnd = random.Random(hashlib.sha256(q["instructions"].encode()).hexdigest())
            err = rnd.random() < 0.1
            out.append({"choice": "error" if err else "correct", "confidence": round(0.5 + rnd.random() / 2, 3)})
        return out

    def ask(self, state, questions):
        payload = build_payload(self.model, state, questions)
        key = hashlib.sha256((self.backend + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            self._log({"cached": True, "tokens_in": 0, "n_questions": len(questions), "s": "labeling"})
            return hit["answers"]
        t0 = time.time()
        if self.backend == "mock":
            answers, tokens, estimated = self._mock(questions), len(json.dumps(payload)) // 4, True
        else:
            body = self._post(payload)
            answers = parse_response(body, questions)
            tokens, estimated = input_tokens_of(body, payload)
        rec = {"k": key, "answers": answers}
        with self.lock:
            self.cache[key] = rec
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        self._log({"cached": False, "tokens_in": tokens, "estimated": estimated, "n_questions": len(questions),
                   "latency_s": round(time.time() - t0, 3), "s": "labeling"})
        return answers
