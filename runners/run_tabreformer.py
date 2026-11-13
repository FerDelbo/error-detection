#!/usr/bin/env python3
"""TabReformer (Nashaat et al., ACM/IMS TDS 2021) no formato comum do pipeline -> pred.csv

  python runners/run_tabreformer.py --name hospital --dirty dataset/hospital/dirty.csv \
         --clean dataset/hospital/clean.csv --budget 20 --seed 0
  saída: outputs/<name>/tabreformer_b<budget>_s<seed>/{pred.csv, metadata.json, labeled_rows.csv, train.csv}

O QUE É REPRODUZIDO
O código publicado (methods/TabReformer) é o BERT original em TensorFlow 1.15 renomeado; o método
em si está nos notebooks examples/Tutorial *.ipynb: BERT-base uncased ajustado (fine-tuning) para
classificar o VALOR de cada célula como correto/errado. Este runner é a mesma rede e o mesmo
treino em PyTorch (TF 1.15 não instala em Python >= 3.8), com os hiperparâmetros dos tutoriais:
  bert-base-uncased (= tfhub bert_uncased_L-12_H-768_A-12), entrada "[CLS] valor [SEP]" com no
  máximo 128 tokens, lote 32, 3 épocas, lr 2e-5 com 10% de aquecimento e queda linear, Adam do
  BERT (sem correção de viés, decaimento 0.01, corte de gradiente 1.0), dropout 0.1 só na saída
  agregada (o tutorial carrega o módulo do hub sem a tag "train": dropout interno desligado).
O pré-treino "Masked Data Model" do artigo (seção 3.2.1) NÃO está no código publicado nem é
descrito o bastante para ser refeito; como nos tutoriais, parte-se do BERT pré-treinado.

DE ONDE VÊM OS RÓTULOS (--protocol)
  paper (padrão): protocolo do artigo (seções 3.2.2, 3.2.3 e 4.1), sem vazamento. Sorteia
      --budget tuplas; só nelas o clean.csv é consultado (rotulador simulado, como no Raha).
      Essas células formam o D_t; a augmentação (--augment) cria erros sintéticos a partir dos
      valores corretos do D_t até a razão corretos/errados chegar a --gamma.
  released: geração de treino dos tutoriais (create_training_data): uma célula aleatória por
      tupla do clean.csv INTEIRO, metade corrompida com 4 transformações fixas; 80% treina.
      Usa o clean.csv todo -> não é comparável com métodos de orçamento limitado; serve para
      conferir o porte contra os números dos tutoriais (extra.heldout_sintetico no metadata).

AUGMENTAÇÃO (--augment, só no protocolo paper)
  learned (padrão): algoritmos 1 e 2 e equações 7-9 do artigo — as transformações são extraídas
      dos pares (valor correto, valor observado) das células erradas do D_t e aplicadas aos
      valores corretos. Se o D_t não tiver nenhum erro, usa as genéricas (registrado no metadata).
  generic: as 4 transformações dos tutoriais (embaralhar, apagar, letra no início/fim).
  none: variante TabReformer_Supervised do artigo (só reamostra os erros do D_t).

A predição é a saída do modelo para todas as células, inclusive as das tuplas rotuladas;
labeled_rows.csv permite avaliá-las à parte (evaluate.py --exclude-rows).
O artigo reporta a média de 15 execuções: varie --seed.
"""
import argparse
import math
import os
import random
import string
import sys
import time
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_meta import guard_output, write_metadata  # noqa: E402

# hiperparâmetros dos tutoriais (examples/Tutorial *.ipynb)
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1


# ---------------------------------------------------------------- augmentação
def generic_corrupt(value, rng):
    """As 4 transformações de create_training_data (tutoriais). Devolve None se o valor não muda."""
    for _ in range(8):
        x = rng.randint(0, 3)
        if x == 0:
            new = "".join(rng.sample(value, len(value)))
        elif x == 1:
            new = value + rng.choice(string.ascii_letters)
        elif x == 2:
            new = ""
        else:
            new = rng.choice(string.ascii_letters) + value
        if new != value:  # o tutorial rotula como erro mesmo sem mudança (ex.: apagar um vazio)
            return new
    return None


def _similarity(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def extract_transformations(pairs):
    """Algoritmo 1: multiconjunto F de transformações (trecho correto -> trecho observado)."""
    found = []
    stack = [p for p in pairs]
    while stack:
        clean, dirty = stack.pop()
        if clean == dirty:
            continue
        found.append((clean, dirty))
        m = SequenceMatcher(None, clean, dirty, autojunk=False).find_longest_match(
            0, len(clean), 0, len(dirty))
        if m.size == 0:
            continue
        cl, cr = clean[:m.a], clean[m.a + m.size:]
        dl, dr = dirty[:m.b], dirty[m.b + m.size:]
        if _similarity(cl, dl) + _similarity(cr, dr) > _similarity(cl, dr) + _similarity(cr, dl):
            stack += [(cl, dl), (cr, dr)]
        else:
            stack += [(cl, dr), (cr, dl)]
    return found


def learned_corrupt(value, prob, rng):
    """Equações 8 e 9: sorteia uma transformação aplicável a `value` e a aplica."""
    cand = [f for f in prob if f[0] in value]
    if not cand:
        return None
    p = np.array([prob[f] for f in cand])
    norm = p.max() - p.min()
    w = (p - p.min()) / norm if norm > 0 else np.ones(len(cand))  # norm = 0: todas iguais
    src, dst = cand[rng.choices(range(len(cand)), weights=w)[0]]
    if src == "":
        pos = rng.randint(0, len(value))
    else:
        starts = [i for i in range(len(value) - len(src) + 1) if value.startswith(src, i)]
        pos = rng.choice(starts)
    new = value[:pos] + dst + value[pos + len(src):]
    return new if new != value else None


# ---------------------------------------------------------------- dados de treino
def parse_budget(text, n_rows):
    text = text.strip()
    if text.endswith("%"):
        k = int(round(n_rows * float(text[:-1]) / 100))
    else:
        k = int(text)
    if not 1 <= k <= n_rows:
        sys.exit(f"--budget inválido: {text!r} dá {k} tuplas (a tabela tem {n_rows})")
    return k


def build_paper_train(dirty, clean, k, augment, gamma, seed):
    """D_t = células das k tuplas sorteadas; depois a augmentação. O clean só é lido nessas tuplas."""
    rng = random.Random(seed)
    rows = sorted(np.random.RandomState(seed).choice(len(dirty), size=k, replace=False).tolist())
    d, c = dirty.values[rows], clean.values[rows]
    wrong = d != c
    examples = [(v, int(w), "rotulado") for v, w in zip(d.ravel(), wrong.ravel())]
    correct = [v for v, w in zip(d.ravel(), wrong.ravel()) if not w]
    n_err = int(wrong.sum())
    target = int(math.ceil(len(correct) / gamma))
    info = {"tuplas_rotuladas": k, "celulas_rotuladas": int(d.size), "erros_rotulados": n_err,
            "alvo_erros": target, "augment_usado": augment}

    if augment == "none":
        errors = [e for e in examples if e[1] == 1]
        if errors and n_err < target:
            examples += [(v, 1, "reamostrado") for v, _, _ in rng.choices(errors, k=target - n_err)]
    elif n_err < target and correct:
        prob = {}
        if augment == "learned":
            F = extract_transformations(list(zip(c[wrong], d[wrong])))
            counts = Counter(F)
            prob = {f: n / len(F) for f, n in counts.items()}  # equação 7
            info["transformacoes"] = len(prob)
            if not prob:
                print("[tabreformer] AVISO: nenhuma célula errada no D_t; sem transformações para "
                      "aprender, usando as genéricas dos tutoriais")
                info["augment_usado"] = "generic"
        made, tries = 0, 0
        while made < target - n_err and tries < 50 * target:
            tries += 1
            src = rng.choice(correct)
            new = learned_corrupt(src, prob, rng) if prob else generic_corrupt(src, rng)
            if new is not None:
                examples.append((new, 1, "sintetico"))
                made += 1
        info["erros_sinteticos"] = made
        if made < target - n_err:
            print(f"[tabreformer] AVISO: só {made} de {target - n_err} erros sintéticos gerados "
                  "(poucas transformações aplicáveis)")
    return pd.DataFrame(examples, columns=["value", "label", "origem"]), rows, info


def build_released_train(clean, seed):
    """create_training_data + load_dataset dos tutoriais: 1 célula por tupla do clean, 80/20."""
    from sklearn.model_selection import train_test_split
    rng = random.Random(seed)
    examples = []
    for i, row in enumerate(clean.values):
        value = row[rng.randint(0, len(row) - 1)]
        new = generic_corrupt(value, rng) if i % 2 == 0 else None
        examples.append((value, 0, "clean") if new is None else (new, 1, "sintetico"))
    data = pd.DataFrame(examples, columns=["value", "label", "origem"])
    train, test = train_test_split(data, test_size=0.2, random_state=42)  # como no tutorial
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------- modelo (porte dos tutoriais)
def make_optimizer(model, torch):
    """AdamWeightDecayOptimizer do BERT (tabReformer/optimization.py): sem correção de viés."""

    class BertAdam(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, dict(lr=0.0, weight_decay=0.0))

        @torch.no_grad()
        def step(self):
            for g in self.param_groups:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if not st:
                        st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    st["m"].mul_(0.9).add_(p.grad, alpha=0.1)
                    st["v"].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                    update = st["m"] / (st["v"].sqrt() + 1e-6)
                    if g["weight_decay"]:
                        update = update + g["weight_decay"] * p
                    p.add_(update, alpha=-g["lr"])

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if "LayerNorm" in name or "bias" in name else decay).append(p)
    return BertAdam([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}])


def load_model(name, seed):
    import torch
    from transformers import BertForSequenceClassification, BertTokenizerFast
    torch.manual_seed(seed)
    tok = BertTokenizerFast.from_pretrained(name)
    model = BertForSequenceClassification.from_pretrained(
        name, num_labels=2, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        classifier_dropout=1 - 0.9)  # keep_prob=0.9 do tutorial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model.to(device), device


def encode(tok, values, device):
    # preenchimento só até o maior do lote: com a máscara de atenção dá a mesma saída que 128 fixo
    enc = tok(list(values), truncation=True, max_length=MAX_SEQ_LENGTH, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def train(tok, model, device, values, labels, seed):
    import torch
    n = len(values)
    steps = int(n / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    warmup = int(steps * WARMUP_PROPORTION)
    if steps < 1:
        sys.exit(f"só {n} exemplos de treino: aumente --budget")
    rs = np.random.RandomState(seed)
    # o tutorial repete os dados e corta lotes de 32 no fluxo contínuo (d.repeat().batch(32))
    stream = np.concatenate([rs.permutation(n) for _ in range(int(math.ceil(steps * BATCH_SIZE / n)))])
    opt = make_optimizer(model, torch)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    model.train()
    for t in range(steps):
        lr = LEARNING_RATE * (t / warmup if t < warmup else 1.0 - t / steps)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = stream[t * BATCH_SIZE:(t + 1) * BATCH_SIZE]
        loss = model(**encode(tok, [values[i] for i in idx], device), labels=y[idx]).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if t % 20 == 0 or t == steps - 1:
            print(f"[tabreformer] passo {t + 1}/{steps} loss={loss.item():.4f}", flush=True)
    return steps


def predict(tok, model, device, values):
    """Classe (1 = erro) de cada valor distinto. Sem dropout (o tutorial o deixa ligado por engano)."""
    import torch
    uniq = sorted(set(values), key=len)  # lotes de tamanhos parecidos: só acelera
    out = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(uniq), 256):
            batch = uniq[i:i + 256]
            pred = model(**encode(tok, batch, device)).logits.argmax(-1).tolist()
            out.update(zip(batch, pred))
    return out


def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(float((y == p).mean()), 4), "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------- linha de comando
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="rótulo do dataset (usado só nos nomes de saída)")
    ap.add_argument("--dirty", required=True, help="caminho do CSV sujo")
    ap.add_argument("--clean", required=True, help="caminho do CSV limpo (rotulador simulado)")
    ap.add_argument("--protocol", choices=["paper", "released"], default="paper",
                    help="paper: orçamento de tuplas + augmentação do artigo (padrão); "
                         "released: treino dos tutoriais, usa o clean.csv inteiro")
    ap.add_argument("--budget", default="20",
                    help="tuplas rotuladas: número (padrão 20, como o Raha) ou porcentagem, ex.: 5%% "
                         "(o artigo usa D_t de 5%% a 15%%)")
    ap.add_argument("--augment", choices=["learned", "generic", "none"], default="learned")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="razão corretos/errados no treino final (os dados dos tutoriais usam 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="bert-base-uncased", help="modelo do Hugging Face ou pasta local")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--overwrite", action="store_true",
                    help="permite sobrescrever uma pasta de saída que tenha resultado de outra configuração")
    args = ap.parse_args()

    dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
    for p in (dirty_path, clean_path):
        if not os.path.isfile(p):
            sys.exit(f"arquivo não encontrado: {p}")
    if args.gamma <= 0:
        sys.exit("--gamma deve ser positivo")
    load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)  # noqa: E731
    dirty, clean = load(dirty_path), load(clean_path)
    if dirty.shape != clean.shape:
        sys.exit(f"tamanhos diferentes: dirty{dirty.shape} clean{clean.shape}")

    # o nome da pasta inclui todos os parâmetros que mudam o resultado, para não sobrescrever
    config = {"protocol": args.protocol, "seed": args.seed, "model": args.model,
              "max_seq_length": MAX_SEQ_LENGTH, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
              "epochs": NUM_TRAIN_EPOCHS, "warmup": WARMUP_PROPORTION}
    if args.protocol == "paper":
        k = parse_budget(args.budget, len(dirty))
        method = "tabreformer_b" + args.budget.strip().replace("%", "pct")
        if args.augment != "learned":
            method += "_" + args.augment
        if args.gamma != 1.0:
            method += f"_g{args.gamma:g}"
        config.update(budget=args.budget.strip(), augment=args.augment, gamma=args.gamma)
    else:
        method = "tabreformer_released"
    if args.model != "bert-base-uncased":
        method += "_" + os.path.basename(args.model.rstrip("/"))
    method += f"_s{args.seed}"
    out = os.path.join(args.outdir, args.name, method)
    guard_output(out, config, args.overwrite)

    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    extra, rows, heldout = {}, [], None
    if args.protocol == "paper":
        train_df, rows, extra = build_paper_train(dirty, clean, k, args.augment, args.gamma, args.seed)
    else:
        train_df, heldout = build_released_train(clean, args.seed)
    n_pos = int(train_df.label.sum())
    print(f"[tabreformer] treino: {len(train_df)} exemplos ({n_pos} errados, {len(train_df) - n_pos} corretos)")
    if n_pos in (0, len(train_df)):
        print("[tabreformer] AVISO: treino com uma classe só; o modelo vai prever sempre a mesma classe")

    tok, model, device = load_model(args.model, args.seed)
    steps = train(tok, model, device, train_df.value.tolist(), train_df.label.tolist(), args.seed)
    cells = dirty.values.ravel().tolist()
    wanted = cells + (heldout.value.tolist() if heldout is not None else [])
    pred = predict(tok, model, device, wanted)
    mat = np.array([pred[v] for v in cells], dtype=int).reshape(dirty.shape)
    elapsed = time.time() - t0

    if heldout is not None:  # mesma conta do estimator.evaluate() dos tutoriais
        extra["heldout_sintetico"] = prf(heldout.label.values, [pred[v] for v in heldout.value])
        print("[tabreformer] 20% sintético separado (comparável aos tutoriais):", extra["heldout_sintetico"])
    extra.update(exemplos_treino=len(train_df), exemplos_errados=n_pos, passos=steps, device=str(device))

    os.makedirs(out, exist_ok=True)
    pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
    train_df.to_csv(os.path.join(out, "train.csv"), index=False)
    # tuplas cujo rótulo o método viu: avalie também sem elas (evaluate.py --exclude-rows)
    pd.DataFrame({"row": rows}).to_csv(os.path.join(out, "labeled_rows.csv"), index=False)
    write_metadata(out, method, args.name, config, elapsed, extra=extra)
    print(f"[tabreformer] {args.name} {method} detectadas={int(mat.sum())} tempo={elapsed:.1f}s -> {out}/pred.csv")


if __name__ == "__main__":
    main()
