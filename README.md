# Atlas Diagnostics Command Center (Jose & Jhon)

Este paquete es una variante del proyecto final orientada para Jose y Jhon. Mantiene toda la funcionalidad del pipeline original (entrenamiento con `train_models.py`, predicción individual y por lotes) pero incorpora un diseño industrial/dark con narrativa de “centro de comando”.

## Características destacadas
- **Dashboard táctico** con tarjeta de misión, playbook rápido y contadores sincronizados de clases/variables.
- **Modo lote** (`#batch-lab`): acepta `.csv`, `.xlsx` y `.xls`, entrega matriz de confusión en PNG (base64), métricas agregadas (exactitud, precisión, sensibilidad, F1 armónico) y vista previa de los primeros 10 registros.
- **Modo individual** (`#single-lab`): formulario autogenerado con todas las variables predictoras; reporta clase predicha y probabilidades.
- **Referencias operativas** (`#intel`): métricas del entrenamiento almacenadas en `models/metrics.json`.
- **Pipelines reproducibles**: cada modelo persiste en `models/*.joblib` y puede reentrenarse ejecutando `python train_models.py`.

## Estructura
```
final_project_scout/
├── app.py
├── train_models.py
├── requirements.txt
├── data/
│   ├── balanced_normalized_dataset_covid_19_hiv.xlsx
│   └── DEMALE-HSJM_2025_data.xlsx
├── models/
│   ├── logistic_regression.joblib
│   ├── mlp_classifier.joblib
│   └── metrics.json
├── static/
│   ├── app.js
│   └── styles.css
└── templates/
    └── index.html
```

## Uso rápido
1. (Opcional) crear entorno virtual.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Entrenar o actualizar modelos:
   ```bash
   python train_models.py          # usa DEMALE por defecto
   # python train_models.py --dataset covid_hiv  # dataset alterno
   ```
4. Iniciar backend:
   ```bash
   python app.py
   ```
5. Abrir `http://127.0.0.1:5000` en el navegador.

## Columnas obligatorias para modo lote
El archivo debe incluir todas las variables predictoras documentadas en `metrics.json` y la columna objetivo `diagnosis`. El orden de columnas se publica vía `GET /api/status`.

## Endpoints clave
| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/api/status` | Devuelve modelo activos, orden de variables, clases y métricas de referencia. |
| `POST` | `/api/predict/individual` | JSON `{ "model": "...", "features": { ... } }` → clase + probabilidades. |
| `POST` | `/api/predict/batch` | Form-data con `model` y `file` → métricas, matriz de confusión (PNG base64) y vista previa. |

## Notas
- Si se reemplaza el dataset o se modifican hiperparámetros, ejecutar `python train_models.py` para regenerar artefactos.
- El frontend obtiene orden de variables desde el backend; cualquier cambio tras el reentrenamiento se refleja automáticamente.
- Para la sustentación, capturar pantallas del modo lote (matriz de confusión) y del modo individual con probabilidades.
