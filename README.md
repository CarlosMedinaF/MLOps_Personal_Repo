# Adult Income: experimentación con MLflow

Script de entrenamiento parametrizado e instrumentado con **MLflow Tracking** para clasificar si una persona gana más de 50K USD al año (dataset [UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult)).

El modelo es un `Pipeline` de scikit-learn: `StandardScaler` (numéricas) + `OneHotEncoder` (categóricas) → `LogisticRegression` (solver `liblinear`).

Actividad individual de TC5061 Operaciones de aprendizaje automático (MNA, Tec de Monterrey).

## Estructura

```
├── train.py              # carga, limpieza, split, entrenamiento, evaluación y registro en MLflow
├── compare_runs.py       # exporta la tabla de corridas del experimento a CSV
├── requirements.txt      # dependencias con versiones fijas
├── notebooks/
│   └── 01_EDA_adult_income.ipynb   # solo exploración de datos
└── data/                 # se crea solo; adult.csv se descarga de UCI la primera vez
```

## Instalación

Probado con Python 3.13 (Anaconda) en Windows / PowerShell.

```powershell
git clone https://github.com/CarlosMedinaF/MLOps_Personal_Repo.git
cd MLOps_Personal_Repo
pip install -r requirements.txt
```

## Ejecución

1. En una terminal, levanta el servidor de MLflow y déjalo abierto:

```powershell
python -m mlflow ui
```

2. En otra terminal, dentro de la carpeta del repo:

```powershell
python train.py --help    # ver argumentos
python train.py           # corre con valores por defecto
```

3. Abre http://127.0.0.1:5000, entra al experimento `adult_income_logreg_experimentos`, selecciona las corridas y da clic en **Compare**.

## Argumentos

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--C` | `1.0` | Inverso de la regularización (menor = más regularización) |
| `--penalty` | `l2` | Tipo de regularización: `l1` o `l2` |
| `--class-weight` | `none` | `none` o `balanced` (compensa el desbalance de clases) |
| `--max-iter` | `1000` | Máximo de iteraciones del solver |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--seed` | `42` | Semilla para el split y el modelo |
| `--experiment-name` | `adult_income_logreg_experimentos` | Experimento de MLflow |
| `--tracking-uri` | `http://127.0.0.1:5000` | Servidor de MLflow |

## Corridas del experimento

```powershell
python train.py
python train.py --C 0.01
python train.py --C 100
python train.py --penalty l1
python train.py --class-weight balanced
python train.py            # repetición para verificar reproducibilidad
```

Para exportar la tabla de resultados:

```powershell
python compare_runs.py
```

## Qué se registra en cada corrida

- **Parámetros:** `C`, `penalty`, `class_weight`, `max_iter`, `solver`, `test_size`, `seed`, `model`.
- **Métricas:** accuracy, precision, recall, F1 y ROC-AUC en test (`test_*`) y en train (`train_*`).
- **Artefactos:** `confusion_matrix.png` y `metrics.json`.
- **Modelo:** el pipeline completo con `mlflow.sklearn.log_model`.

## Reproducibilidad

- La semilla (`--seed`) fija el `train_test_split` estratificado y el `random_state` del modelo; también se fijan `random` y `numpy`.
- Dos ejecuciones con los mismos argumentos producen las mismas métricas.
- `requirements.txt` fija las versiones exactas de las librerías.
