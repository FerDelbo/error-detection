# ZeroED + Jev (variante)

Variante do ZeroED em que o **Jev** (jevai.net, modelo de decisão/classificação da TypeSafe AI)
rotula os valores centrais no lugar do LLM. A propagação dos rótulos pelos clusters e o
classificador final (MLP por atributo) continuam os do ZeroED.

**Isolamento:** esta pasta é uma cópia congelada de `methods/zeroed/` (com os patches do
`runners/run_zeroed.py` já aplicados). Nada aqui lê ou escreve em `methods/zeroed/`, exceto o
fastText, que é usado por link simbólico e só lido. Toda mudança em relação ao original está
marcada com `# JEV`:

```bash
diff methods/zeroed/main.py methods/zeroed_jev/main.py
diff methods/zeroed/utility.py methods/zeroed_jev/utility.py
```

## Variantes

| Variante | Funções clean (2) | Distribuição/diretrizes (4, 5) | Rotulação (6) | Funções de detecção (8) | Erros sintéticos (9) | Proteção |
|---|---|---|---|---|---|---|
| `base` | LLM | LLM | LLM | LLM | LLM | não |
| `jevlab` | LLM | — | **Jev** | LLM | LLM | não |
| `jevlab_feat` | — | — | **Jev** | LLM | LLM | não |
| `jevlab_aug` | — | — | **Jev** | — | LLM | sim |

- `base` é o ZeroED original rodando com o **mesmo Qwen** das outras variantes. É a referência
  justa: os `zeroed_s<N>` já existentes usaram outro LLM. Serve também para conferir se a cópia
  se comporta como o original.
- Cada passo da tabela isola um efeito: `base→jevlab` mostra o efeito do rotulador,
  `jevlab→jevlab_feat` o das funções da clusterização e `jevlab_feat→jevlab_aug` o das
  features do MLP.
- Com a amostragem `RANDOM` (`np.random.seed(0)`), os centros rotulados são os mesmos em todas
  as variantes. Os rótulos do Jev saem iguais em `jevlab`, `jevlab_feat` e `jevlab_aug` e vêm do cache.

**Proteção (`empty_func_guard`).** No ZeroED original, um atributo sem nenhuma função tem
`pass_num/(0+1e-6) = 0 < 0.5` e perde **todos** os rótulos "correto". O MLP daquele atributo
treina só com exemplos de erro. Em `jevlab_aug` isso aconteceria em todas as colunas, então
nessa variante a filtragem é pulada para atributos sem funções. Nas outras, o comportamento é o original.

## O que muda em cada arquivo

| Arquivo | Mudança |
|---|---|
| `main.py` | `jev.labeler` (llm \| jev), `jev.empty_func_guard`, `set_stage()` antes de cada etapa com LLM, `center_labels.json` gravado em toda execução |
| `utility.py` | LLM pelas variáveis `QWEN_*` (padrões do GIDCL), temperatura `ZEROED_JEV_TEMPERATURE` (padrão 0), remoção de `<think>`, log de uso com a etapa (`"s"`) |
| `column_profile.py` (novo) | perfil da coluna calculado só a partir do dirty, contexto do Jev no lugar das diretrizes |
| `jev_labeler.py` (novo) | lotes de 20 centros → uma requisição (state = perfil + linhas; uma pergunta error/correct por linha) → rótulos montados pelo índice da linha |
| `jev_client.py` (novo) | **único arquivo dependente da API do Jev** (`build_payload`, `parse_response`), com novas tentativas, cache e log de uso |
| `prompt_gen.py`, `feature.py`, `distri_analys.py`, `get_rel_attrs.py`, `measure.py` | cópias sem mudança |

## Como rodar

Variáveis no `.env`:

```bash
# LLM (o mesmo Qwen do GIDCL; descomente só para mudar o padrão)
# QWEN_MODEL=Qwen/Qwen2.5-14B-Instruct
# QWEN_BASE_URL=http://localhost:8000/v1
# QWEN_API_KEY=EMPTY
# Jev
JEV_URL=<endpoint, ver docs.typesafe.ai>
JEV_API_KEY=<chave>
JEV_MODEL=jev-1.14.0
```

Comandos:

```bash
# todas as variantes, 1 execução
ZEROED_PY=~/miniconda3/envs/zeroed/bin/python \
  bash scripts/run_zeroed_jev.sh hospital dataset/hospital/dirty.csv dataset/hospital/clean.csv

# uma variante
VARIANTS=jevlab bash scripts/run_zeroed_jev.sh hospital dataset/hospital/dirty.csv dataset/hospital/clean.csv

# teste do encanamento sem a API do Jev (respostas falsas; saída com sufixo _mock)
JEV_BACKEND=mock VARIANTS=jevlab bash scripts/run_zeroed_jev.sh hospital dataset/hospital/dirty.csv dataset/hospital/clean.csv
```

Saída: `outputs/<nome>/zeroed_<variante>_qwen_s<N>/`. Em `metadata.json → extra`:
- `llm_por_etapa_total`: chamadas e tokens do LLM separados por etapa;
- `jev_total`: requisições, respostas do cache, tokens de entrada e custo estimado;
- `qualidade_rotulos_centros`: P/R/F1 dos rótulos dos centros contra o ground truth (usado só no relatório).

## Pendências

- **Formato da API:** o formato do Jev em `jev_client.py` segue o site (artigo da versão 1.14) e ainda
  **não foi conferido** na documentação oficial. Ao ter acesso, ajuste `build_payload` e `parse_response`.
- **Limiar:** p(erro) > 0.5 (`--threshold`). É fixo; não deve ser ajustado com o ground truth.
