import argparse, os, random, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "methods", "raha-master"))
from detection import Detection
from pipeline_meta import guard_output, write_metadata

# valores aceitos por methods/raha-master/detection.py (ERROR_DETECTION_ALGORITHMS e CLASSIFICATION_MODEL)
ALGORITHMS = ("OD", "PVD", "RVD", "KBVD", "TFIDF")
DEFAULT_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]
CLASSIFIERS = ("GBC", "ABC", "DTC", "GNB", "KNC", "SGDC", "SVC")

ap = argparse.ArgumentParser(description="Raha (detecção) -> pred.csv")
ap.add_argument("--name", required=True, help="rótulo do dataset (usado só nos nomes de saída)")
ap.add_argument("--dirty", required=True, help="caminho do CSV sujo")
ap.add_argument("--clean", required=True, help="caminho do CSV limpo (usado para simular o rotulador)")
ap.add_argument("--budget", type=int, default=20, help="tuplas rotuladas (artigo: 20)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--algorithms", default="OD,PVD,RVD,KBVD",
                help="famílias de detectores separadas por vírgula: OD, PVD, RVD, KBVD, TFIDF "
                     "(padrão do artigo: OD,PVD,RVD,KBVD)")
ap.add_argument("--classifier", default="GBC", choices=CLASSIFIERS,
                help="classificador por coluna (padrão do artigo: GBC)")
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
ap.add_argument("--overwrite", action="store_true",
                help="permite sobrescrever uma pasta de saída que tenha resultado de outra configuração")
args = ap.parse_args()

algorithms = [a.strip().upper() for a in args.algorithms.split(",") if a.strip()]
invalid = [a for a in algorithms if a not in ALGORITHMS]
if not algorithms or invalid:
    sys.exit(f"--algorithms inválido: {invalid or args.algorithms!r}; use {', '.join(ALGORITHMS)}")
algorithms = list(dict.fromkeys(algorithms))  # remove repetidos, mantém a ordem

dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
for p in (dirty_path, clean_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")

# o nome da pasta inclui todos os parâmetros que mudam o resultado, para não sobrescrever
method = f"raha_b{args.budget}_s{args.seed}"
if algorithms != DEFAULT_ALGORITHMS:
    method += "_" + "+".join(algorithms)
if args.classifier != "GBC":
    method += "_" + args.classifier
out = os.path.join(args.outdir, args.name, method)
config = {"budget": args.budget, "seed": args.seed, "algorithms": "+".join(algorithms),
          "classifier": args.classifier}
guard_output(out, config, args.overwrite)

random.seed(args.seed)
np.random.seed(args.seed)

app = Detection()
app.LABELING_BUDGET = args.budget
app.USER_LABELING_ACCURACY = 1.0
app.SAVE_RESULTS = False
app.VERBOSE = False
app.ERROR_DETECTION_ALGORITHMS = algorithms
app.CLASSIFICATION_MODEL = args.classifier

dd = {"name": args.name, "path": dirty_path, "clean_path": clean_path}
t0 = time.time()
detected = app.run(dd)
elapsed = time.time() - t0

print("exemplo de chaves:", list(detected)[:3])
dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
mat = np.zeros(dirty.shape, dtype=int)
for (i, j) in detected:
    mat[i, j] = 1

os.makedirs(out, exist_ok=True)
pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
# tuplas rotuladas pelo "usuário" (rótulo copiado do clean.csv): o Raha devolve o próprio rótulo
# nelas, então avaliá-las infla P/R/F1 conforme o budget cresce -> use --exclude-rows no evaluate.py
labeled_rows = sorted(int(i) for i in app.d.labeled_tuples)
pd.DataFrame({"row": labeled_rows}).to_csv(os.path.join(out, "labeled_rows.csv"), index=False)
write_metadata(out, method, args.name, config, elapsed)
print(f"[raha] {args.name} seed={args.seed} budget={args.budget} algorithms={'+'.join(algorithms)} "
      f"classifier={args.classifier} detectadas={int(mat.sum())} tempo={elapsed:.1f}s -> {out}/pred.csv")