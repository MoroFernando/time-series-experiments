# Dimensionality Reduction Experiments for Time Series Classification

Benchmark que avalia o impacto de diferentes técnicas de **redução de dimensionalidade** na **acurácia de classificadores** de séries temporais de alta dimensionalidade, além de métricas de **preservação de vizinhança** da estrutura original dos dados.

---

## Estrutura do projeto

```
├── src/
│   ├── reduction.py       # Métodos de redução (PAA, DFT, DWT, SVD, PCA, KPCA, Isomap, AE, CAE)
│   ├── classifiers.py     # Registro de classificadores (Rocket, Catch22, QUANT, 1NN-DTW)
│   ├── datasets.py        # Carregamento e Z-normalização dos datasets
│   ├── experiment.py      # Lógica do experimento (redução → métricas → classificação)
│   └── metrics.py         # Preservação de vizinhança (Precision@K, Trustworthiness)
├── results/               # CSVs de saída (criado automaticamente)
├── config.yaml            # Parâmetros do experimento
├── run_experiment.py      # Entrypoint
└── requirements.txt
```

---

## O que o experimento faz

Para cada combinação de `dataset × método de redução × taxa de retenção × classificador`:

1. **Carrega** o dataset via `aeon` e aplica Z-normalização por série
2. **Reduz** as séries de treino e teste ao tamanho `w = comprimento_original × retention_rate`
3. **Calcula** métricas de preservação de vizinhança sobre o conjunto de teste (uma vez por método/taxa) e salva em `results/results_neighborhood.csv`
4. **Treina e avalia** cada classificador, salvando acurácia e tempos em `results/results.csv`

Cada linha é salva imediatamente no CSV — uma falha não perde resultados anteriores.

---

## Datasets disponíveis

Séries temporais de alta dimensionalidade (>= 1000 pontos) do repositório UEA/UCR via `aeon`:

| Dataset | Timepoints | Classes |
|---|---|---|
| ACSF1 | 1460 | 10 |
| CinCECGTorso | 1639 | 4 |
| EOGHorizontalSignal | 1250 | 12 |
| EOGVerticalSignal | 1250 | 12 |
| EthanolLevel | 1751 | 4 |
| HandOutlines | 2709 | 2 |
| Haptics | 1092 | 5 |
| HouseTwenty | 3000 | 2 |
| InlineSkate | 1882 | 7 |
| Mallat | 1024 | 8 |
| MixedShapesRegularTrain | 1024 | 5 |
| MixedShapesSmallTrain | 1024 | 5 |
| Phoneme | 1024 | 39 |
| PigAirwayPressure | 2000 | 52 |
| PigArtPressure | 2000 | 52 |
| PigCVP | 2000 | 52 |
| Rock | 2844 | 4 |
| SemgHandGenderCh2 | 1500 | 2 |
| SemgHandMovementCh2 | 1500 | 6 |
| SemgHandSubjectCh2 | 1500 | 5 |
| StarLightCurves | 1024 | 3 |

---

## Pré-requisitos

- **Python 3.10+** (obrigatório — o código usa sintaxe de type hints do Python 3.10)
- **Conda** (recomendado para gerenciar o ambiente)
- Acesso à internet para download automático dos datasets via `aeon`

---

## Instalação passo a passo

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd dimensionality-reduction-experiments
```

### 2. Criar o ambiente conda

```bash
conda create -n pibic python=3.10 -y
conda activate pibic
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

## Configuração

Edite o `config.yaml` antes de rodar. Comente (`#`) ou descomente as entradas desejadas:

```yaml
datasets:
  - Rock
  # - StarLightCurves

classifiers:
  - Rocket
  - Catch22
  - QUANT
  - 1NN-DTW

reduction_methods:
  - PAA
  - CAE
  # - DFT, DWT, SVD, PCA, KPCA, Isomap, AE

retention_rates:
  - 0.85   # mantém 85% dos pontos
  - 0.50
  - 0.25

output:
  results_file: results/results.csv
  neighborhood_file: results/results_neighborhood.csv
  neighborhood_k: 5   # k para Precision@K e Trustworthiness

reproducibility:
  random_state: 1
```

---

## Execução

### Localmente

```bash
conda activate pibic
python run_experiment.py
```

Para usar um arquivo de configuração alternativo:

```bash
python run_experiment.py --config meu_config.yaml
```

### Em máquina remota via SSH

Experimentos longos devem rodar em sessões persistentes para não serem interrompidos se a conexão SSH cair.

**Opção 1 — tmux** (recomendado: permite reconectar e acompanhar o output):

```bash
tmux new -s experimento
conda activate pibic
python run_experiment.py

# Para desconectar sem matar o processo: Ctrl+B, depois D
# Para reconectar depois: tmux attach -s experimento
```

**Opção 2 — nohup** (simples, redireciona tudo para arquivo):

```bash
nohup python run_experiment.py > experimento.log 2>&1 &

# Acompanhar em tempo real:
tail -f experimento.log

# Ver se ainda está rodando:
jobs
```

---

## Resultados

Dois CSVs são gerados incrementalmente em `results/`:

**`results.csv`** — um registro por classificador:

| Coluna | Descrição |
|---|---|
| `dataset` | Nome do dataset |
| `classifier` | Nome do classificador |
| `reduction_method` | Método de redução (`None` = original) |
| `retention_rate` | Fração de pontos retidos (`None` = original) |
| `series_size` | Número de pontos após redução |
| `accuracy` | Acurácia no conjunto de teste |
| `train_time_s` | Tempo de treinamento (segundos) |
| `test_time_s` | Tempo de inferência (segundos) |
| `reduction_time_s` | Tempo de redução do dataset (segundos) |

**`results_neighborhood.csv`** — um registro por método/taxa (independente de classificador):

| Coluna | Descrição |
|---|---|
| `dataset` | Nome do dataset |
| `reduction_method` | Método de redução |
| `retention_rate` | Fração de pontos retidos |
| `series_size` | Número de pontos após redução |
| `precision@5` | Fração de vizinhos preservados (k=5) |
| `trustworthiness` | Grau de confiança da estrutura local reduzida |
