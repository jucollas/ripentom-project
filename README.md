# RipenTom — Sistema Inteligente de Clasificación de Tomates por Estado de Madurez

**Juliana Sánchez Gómez · Juan Diego Collazos Mejía · Ángel Luis Obando Fajardo · Juan José Torres Murillo · Marco Riascos Salguero**  
Programa de Ingeniería de Sistemas y Computación — Pontificia Universidad Javeriana Cali

---

## 1. Descripción del Problema

La clasificación manual del estado de madurez de frutas y hortalizas en la cadena agroindustrial conlleva costos operativos elevados, alta variabilidad subjetiva entre operarios y riesgos sanitarios asociados al procesamiento de producto en mal estado. El tomate es especialmente sensible a las condiciones de manejo postcosecha, lo que hace crítica una inspección confiable.

**RipenTom** aborda este problema clasificando automáticamente tomates en tres estados:

| Clase | Descripción |
|---|---|
| `ripe` (maduro) | Tomate en su punto óptimo de consumo |
| `unripe` (inmaduro) | Tomate que no ha completado su maduración |
| `damaged` (dañado) | Tomate con defectos superficiales, golpes o pudrición |

El sistema opera a partir de una imagen del tomate, devolviendo la clase predicha y las probabilidades asociadas a cada categoría. La inferencia está disponible como una aplicación web de arrastrar y soltar, desplegada con un frontend estático en Vercel y un backend Flask-API.

### Dataset

El conjunto de datos base fue obtenido de Kaggle y sometido a un protocolo de curación manual sistemático: revisión imagen por imagen, corrección de etiquetas erróneas, eliminación de duplicados y de ejemplos ambiguos. Tras la curación, el desbalanceo residual se corrigió mediante aumentación controlada (rotación ±15°, volteo horizontal/vertical, ajuste de brillo y contraste), obteniendo un dataset final balanceado:

| Clase | Total | Train (70 %) | Val (30 %) |
|---|---|---|---|
| Ripe (maduro) | 400 | 280 | 120 |
| Unripe (inmaduro) | 400 | 280 | 120 |
| Damaged (dañado) | 400 | 280 | 120 |
| **Total** | **1 200** | **840** | **360** |

La división train/val se realizó con `random.seed(42)` para garantizar reproducibilidad.

---

## 2. Arquitectura del Agente

RipenTom está compuesto por dos componentes desacoplados: un **backend de inferencia** expuesto como API REST y un **frontend web estático** que consume dicha API.

```
[Usuario sube imagen]
        |
        v
[Frontend HTML/JS — Vercel]
        |  POST /predict (multipart)
        v
[Flask API Backend]
  ├─ Preprocesamiento (crop por fondo negro, resize 224×224)
  ├─ Extracción de features (100 dimensiones)
  ├─ Normalización (StandardScaler)
  └─ Inferencia (Random Forest, 200 árboles)
        |
        v
[JSON: { clase, probabilidades }]
        |
        v
[Frontend muestra resultado + confianza por clase]
```

### Backend — Flask API

Implementado en Python con Flask. Expone el endpoint `POST /predict` que:

1. Recibe una imagen como archivo multipart.
2. Detecta el objeto segmentando píxeles con intensidad > 10 en cualquier canal.
3. Recorta el bounding box y redimensiona a 224×224.
4. Extrae un vector de **100 características** (ver sección 3).
5. Normaliza con el `StandardScaler` guardado durante el entrenamiento.
6. Invoca el modelo **Random Forest** serializado (200 estimadores).
7. Retorna clase predicha y probabilidades por clase en formato JSON.

El modelo y el scaler se cargan desde archivos `.joblib` al iniciar el servidor.

### Frontend — HTML estático

Aplicación de página única desarrollada con HTML, CSS y JavaScript vanilla (sin frameworks). Permite al usuario:

- Arrastrar y soltar una imagen o seleccionarla desde el dispositivo.
- Previsualizar la imagen antes de enviarla.
- Ver la clasificación resultante y el porcentaje de confianza por clase.

La interfaz muestra explícitamente las probabilidades para cada categoría y recomienda confirmación humana en casos de baja confianza. El sistema no almacena las imágenes enviadas por los usuarios.

### Pipeline de extracción de características

El vector de características de 100 dimensiones se construye a partir de los siguientes grupos:

| Grupo | Descripción | Dimensiones |
|---|---|---|
| Estadísticas HSV | Media, desv. estándar, percentil 25 y 75 por canal (H, S, V) | 12 |
| Estadísticas LAB | Media, desv. estándar, percentil 25 y 75 por canal (L, A, B) | 12 |
| Histogramas HSV | 16 bins para H; 16 bins para S; 16 bins para V (normalizados) | 48 |
| GLCM | Contraste, correlación, energía, homogeneidad, disimilaridad (media y std, distancias [1,2] × ángulos [0°, 45°]) | 10 |
| LBP | Histograma de 10 bins de patrones binarios locales (radio=1, P=8) | 10 |
| Descriptores de defectos | Razón de puntos oscuros (V<50), razón de manchas marrones, número de regiones marrones | 3 |
| Densidad de bordes | Proporción de píxeles Canny sobre el área del objeto | 1 |
| Shape | Área, perímetro, circularidad, aspect ratio del contorno principal | 4 |
| **Total** | | **100** |

Se implementó también un **Enfoque 2** que divide la imagen en 4 bloques (cuadrícula 2×2), extrae el mismo vector base por bloque, promedia los 4 descriptores locales y concatena el resultado al vector global, produciendo un vector de **200 dimensiones** con información espacial local.

---

## 3. Justificación Técnica

### ¿Por qué ML clásico sobre un vector de características artesanal?

El dataset de 1 200 imágenes se encuentra en el rango donde las redes neuronales profundas no logran materializar su capacidad representacional. Los experimentos confirmaron que **todos los modelos de ML clásico superaron a MobileNetV3** en este dominio (accuracy 0.819–0.867 vs 0.783). La ingeniería de características permite incorporar conocimiento experto del dominio (propiedades colorimétricas del proceso de maduración, indicadores de daño superficial) que el modelo de DL tendría que inferir sin suficientes datos.

### ¿Por qué Random Forest como modelo de producción?

Aunque SVM-RBF con cuadrantes alcanza el mejor rendimiento técnico (acc=0.867), Random Forest con el vector de 100 features fue seleccionado para despliegue por:

- **Velocidad de inferencia:** el Enfoque 2 duplica el tiempo de extracción de características, afectando la latencia de la API.
- **Diferencia marginal:** la ganancia de SVM+cuadrantes sobre RF+imagen completa es de ~1.5 puntos porcentuales.
- **Probabilidades calibradas:** `predict_proba` de Random Forest enriquece la respuesta JSON con confianzas por clase, mientras que SVM requiere `probability=True` con calibración adicional.
- **Robustez al ruido:** el ensamble de 200 árboles reduce el sobreajuste respecto a modelos individuales.

### ¿Por qué MobileNetV3-Small para el componente de Deep Learning?

Diseñada específicamente para entornos de recursos limitados, MobileNetV3-Small ofrece un balance adecuado entre eficiencia computacional y capacidad representacional. Su preentrenamiento en ImageNet provee representaciones de bajo nivel (bordes, texturas, formas) transferibles al dominio de inspección de frutas. Se empleó un entrenamiento en **dos fases** para evitar el olvido catastrófico:

1. **Fase 1 (5 épocas):** backbone congelado, solo se entrena el clasificador (lr=1e-3). Permite que las capas nuevas converjan sin distorsionar las representaciones aprendidas.
2. **Fase 2 (10 épocas):** se descongelan los bloques 10+ del backbone y se entrena con lr=5e-5, adaptando las características de alto nivel al dominio específico de tomates.

### ¿Por qué el análisis por cuadrantes 2×2?

Los defectos superficiales en tomates son localizados, no distribuidos uniformemente. El procesamiento global de la imagen tiende a enmascarar irregularidades al promediar regiones sanas y dañadas. La división 2×2 permite capturar distribuciones espaciales distintivas —especialmente para la clase `unripe`, cuya coloración verde uniforme por cuadrante es muy discriminativa— sin incrementar excesivamente la dimensionalidad.

### ¿Por qué la curación manual del dataset es el factor dominante?

El dataset original de Kaggle presentaba etiquetas erróneas, duplicados y ejemplos ambiguos. La corrección sistemática de estos problemas tuvo mayor impacto en el rendimiento final que cualquier decisión de arquitectura. Este hallazgo valida la literatura especializada: errores sistemáticos de etiquetado pueden degradar la accuracy hasta en un 15% incluso con arquitecturas sofisticadas.

---

## 4. Metodología de Evaluación

### Protocolo experimental

- **División:** 70 % entrenamiento / 30 % validación, estratificada por clase, con semilla fija (`random.seed(42)`).
- **Preprocesamiento:** todos los modelos operan sobre el mismo conjunto de imágenes preprocesadas (crop por umbralización, resize 224×224).
- **Normalización:** los features fueron estandarizados (media cero, varianza unitaria) antes de alimentar SVM y KNN, que son sensibles a la escala.

### Métricas

- **Accuracy global:** fracción de predicciones correctas sobre el conjunto de validación (360 muestras).
- **F1-macro:** promedio no ponderado del F1-score por clase, relevante ante posibles desbalanceos residuales.
- **Reporte por clase:** precisión, recall y F1 individuales para `damaged`, `unripe` y `ripe`.
- **Matriz de confusión:** visualización de los patrones de error entre clases.

### Resultados

**Enfoque 1 — Imagen completa (100 features):**

| Modelo | Accuracy | F1-macro | Tiempo |
|---|---|---|---|
| Random Forest | **0.853** | **0.853** | 1.1 s |
| Gradient Boosting | 0.850 | 0.850 | 5.9 s |
| SVM (RBF) | 0.847 | 0.845 | 0.1 s |
| KNN (k=5) | 0.819 | 0.817 | 0.0 s |
| MobileNetV3 (DL) | 0.783 | 0.782 | — |

**Enfoque 2 — Cuadrantes 2×2 (200 features):**

| Modelo | Accuracy | F1-macro | Tiempo |
|---|---|---|---|
| SVM (RBF) | **0.867** | **0.865** | 0.1 s |
| Random Forest | 0.856 | 0.856 | 1.7 s |
| Gradient Boosting | 0.847 | 0.847 | 12.6 s |
| KNN (k=5) | 0.822 | 0.820 | 0.1 s |
| MobileNetV3 (DL) | 0.783 | 0.782 | — |

**Reporte por clase del modelo en producción (Random Forest, Enfoque 1):**

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Damaged | 0.89 | 0.85 | 0.87 | 120 |
| Unripe | 0.89 | 0.90 | 0.89 | 120 |
| Ripe | 0.79 | 0.81 | 0.80 | 120 |
| **Macro avg** | **0.85** | **0.85** | **0.85** | **360** |

### Análisis de riesgos de clasificación

La clase `damaged` presenta el recall más bajo en la mayoría de los modelos (0.66 en MobileNetV3, 0.82–0.85 en ML clásico), lo que indica que el sistema tiende a no detectar algunos casos de daño real. En contextos de alta exigencia sanitaria, este riesgo debe complementarse con inspección humana para muestras de baja confianza. El sistema expone probabilidades por clase precisamente para permitir implementar umbrales de confianza configurables.

---

## 5. Declaración de IA Generativa

En el desarrollo de este proyecto se utilizaron herramientas de inteligencia artificial generativa como apoyo en tareas específicas:

- **Depuración de código:** sugerencias de implementación para la pipeline de extracción de características en Python.
- **Consulta de documentación técnica:** resolución de dudas sobre el uso de bibliotecas como scikit-learn, PyTorch y OpenCV.
- **Revisión de texto:** corrección gramatical y ortográfica del informe escrito en español.

Todo el diseño conceptual del sistema, la toma de decisiones sobre arquitectura, la curación del dataset, la interpretación de resultados y las conclusiones son producto del trabajo original del equipo. Las herramientas de IA fueron utilizadas como apoyo instrumental, no como sustituto del juicio crítico y la comprensión técnica del equipo.

El sistema resultante —RipenTom— opera como herramienta de **apoyo a la decisión**, no como agente autónomo. La clasificación producida es una recomendación probabilística; cualquier acción subsiguiente (descartar un lote, enrutar producto, informar a un operario) recae en un ser humano que interpreta el resultado. Esta separación entre inferencia y actuación es una decisión de diseño deliberada para mantener la supervisión humana en el lazo de control.

---# ripentom-project
