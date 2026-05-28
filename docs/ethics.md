# RipenTom — Consideraciones Éticas

El desarrollo de sistemas basados en inteligencia artificial y visión por computadora para aplicaciones agrícolas representa una herramienta prometedora para optimizar procesos de clasificación y monitoreo de alimentos. Sin embargo, su implementación implica consideraciones éticas que van más allá de la funcionalidad técnica: confiabilidad, precisión, impacto social y uso responsable de la tecnología son aspectos que deben analizarse con la misma rigurosidad que el rendimiento del modelo.

---

## 1. Decisiones — ¿Qué autonomía tiene el sistema?

RipenTom opera como un **sistema de apoyo a la decisión**, no como un agente autónomo con capacidad de acción directa sobre procesos físicos. La autonomía del sistema se limita estrictamente a:

- Recibir una imagen proporcionada por el usuario.
- Emitir una clasificación (`ripe` / `unripe` / `damaged`) junto con las probabilidades asociadas a cada clase.

Cualquier acción subsiguiente —descartar un lote, enrutar producto hacia procesamiento, informar a un operario— recae en un ser humano que interpreta el resultado. Esta separación explícita entre **inferencia y actuación** es una decisión de diseño deliberada para mantener la supervisión humana en el lazo de control.

El sistema no almacena imágenes enviadas por los usuarios, no actúa sobre bases de datos externas ni modifica ningún proceso de forma automática. La interfaz muestra explícitamente las probabilidades por clase y comunica la naturaleza estadística de la predicción, con el objetivo de desincentivar la sobreconfianza en el resultado. En casos de baja confianza, la interfaz recomienda activamente la confirmación humana.

No se debe generar una dependencia total de las tecnologías automatizadas. Aunque RipenTom permite aumentar la eficiencia y reducir tiempos de procesamiento, sigue siendo necesaria la supervisión de operadores o expertos que validen los resultados en entornos reales. El objetivo es mantener un equilibrio entre la automatización de procesos y la supervisión humana: aprovechar las ventajas de la inteligencia artificial sin ceder la capacidad de análisis y toma de decisiones a un sistema que, por definición, trabaja con aproximaciones estadísticas.

---

## 2. Sesgos — ¿Qué prejuicios pueden emerger de los datos?

El desempeño del modelo depende directamente de la calidad, cantidad y diversidad del dataset empleado. El dataset curado de RipenTom, aunque mejorado respecto al original de Kaggle, puede contener sesgos no identificados que limiten su capacidad de generalización:

**Sesgo de dominio (condiciones de captura)**  
Las imágenes del dataset fueron capturadas con fondo negro y condiciones de iluminación relativamente uniformes. En entornos reales —con luz natural variable, fondos complejos o cámaras de menor calidad— el rendimiento del sistema puede degradarse significativamente. Las variaciones naturales de color, tamaño y forma de los frutos en campo pueden diferir considerablemente de las imágenes de laboratorio sobre las que fue entrenado el modelo.

**Sesgo de variedad**  
El modelo fue entrenado sobre imágenes de variedades específicas de tomate. Podría generalizar mal a variedades con formas o coloraciones inusuales que no estuvieran representadas durante el entrenamiento.

**Sesgo de representación del daño**  
La categoría `damaged` engloba tipos heterogéneos de defectos: golpes, pudrición, enfermedades foliares, manchas superficiales. La representación de cada tipo de daño en el dataset puede no ser proporcional a su prevalencia real en campo. Esto implica que el sistema puede desempeñarse de forma desigual ante distintos tipos de defecto dentro de la misma clase.

**Sesgo de etiquetado residual**  
A pesar de la curación manual, el proceso de revisión imagen por imagen no garantiza la eliminación total de errores de etiquetado. La categoría `damaged` fue la más afectada por ambigüedades en el dataset original y la que presentó el recall más bajo en todos los modelos evaluados.

Es importante aclarar que los resultados generados por el sistema corresponden a una **aproximación basada en patrones aprendidos**, no a una evaluación infalible. Estos sesgos son inherentes a cualquier sistema de aprendizaje automático entrenado con datos históricos y deben ser considerados explícitamente al evaluar la idoneidad del sistema para un contexto de despliegue específico.

---

## 3. Riesgos — ¿Qué consecuencias existen si el agente falla?

Un error del sistema en contexto agroindustrial puede tener consecuencias asimétricas según su dirección:

**Falso negativo sobre daño (clasificar `damaged` como `ripe` o `unripe`)**  
Producto en mal estado alcanza al consumidor o continúa en la cadena de distribución. Este escenario implica riesgos sanitarios potenciales y compromete la seguridad alimentaria. Es el tipo de error de mayor gravedad desde una perspectiva de salud pública.

**Falso positivo sobre daño (clasificar `ripe` como `damaged`)**  
Producto en buenas condiciones es descartado innecesariamente. Esto genera pérdidas económicas directas para el productor y reduce la confiabilidad percibida del sistema.

El análisis de los reportes de clasificación evidencia que la clase `damaged` presenta el recall más bajo en todos los modelos evaluados (0.66 en MobileNetV3, entre 0.82 y 0.85 en ML clásico). Esto significa que el sistema tiene una **tendencia a no detectar algunos casos de daño real**, priorizando precisión sobre recall en esa clase. En contextos de alta exigencia sanitaria, este riesgo debe ser:

- Comunicado explícitamente a los operadores del sistema.
- Gestionado mediante umbrales de confianza configurables: derivar a revisión humana toda predicción cuya probabilidad no supere un umbral definido por el usuario.
- Complementado con inspección humana para las muestras de baja confianza, especialmente ante la sospecha de daño.

Adicionalmente, los riesgos se extienden a la confiabilidad operacional: una falla técnica del sistema (caída del servidor, errores de preprocesamiento, imágenes de baja calidad) puede interrumpir procesos de clasificación sin que el operario reciba una señal de advertencia clara. El diseño del sistema debe contemplar mecanismos de fallback y mensajes de error informativos que preserven la supervisión humana incluso ante fallos técnicos.

---

## 4. Impacto — ¿Cómo afecta el despliegue a los humanos?

**Impacto en los usuarios directos**  
Al ser una herramienta web de acceso público, usuarios sin formación técnica pueden someter imágenes a clasificación y percibir el resultado como definitivo. Para mitigar el riesgo de sobreconfianza, la interfaz comunica la naturaleza probabilística de la predicción y no presenta el resultado como un veredicto absoluto. El sistema tampoco almacena las imágenes enviadas, respetando la privacidad de los datos del usuario.

**Impacto sobre los trabajadores agrícolas**  
La automatización de la clasificación de calidad puede reducir parcialmente la participación humana en tareas manuales de inspección. Sin embargo, RipenTom no está diseñado para reemplazar puestos de trabajo, sino para apoyar y agilizar la inspección manual, permitiendo que los operarios concentren su atención en los casos de mayor ambigüedad. Es responsabilidad de quienes adopten el sistema gestionar este impacto de forma justa para los trabajadores afectados, evitando que la herramienta sea utilizada como justificación para reducción de personal sin una transición acompañada.

**Impacto sobre la seguridad alimentaria**  
Cuando el sistema funciona correctamente, ofrece beneficios concretos: procesos de inspección más rápidos, constantes y replicables, con menor variabilidad subjetiva entre operarios. Esto puede fortalecer los mecanismos de control de calidad dentro de la cadena de producción agrícola y contribuir a reducir el riesgo de que producto en mal estado alcance al consumidor.

**Impacto sobre la equidad en el sector agrícola**  
La adopción de tecnologías de inteligencia artificial puede contribuir a la modernización del sector agrícola. Sin embargo, su implementación puede generar brechas tecnológicas entre grandes productores —con infraestructura y recursos para integrar estas herramientas— y pequeños agricultores con limitaciones económicas o de conectividad. Es importante que el despliegue sea accesible y responsable, evitando que la automatización profundice desigualdades preexistentes en el sector.

---

## Reflexión final

RipenTom es una herramienta técnicamente funcional, pero su valor real depende de cómo se use. La responsabilidad profesional en el diseño de sistemas inteligentes implica anticipar no solo los casos en que el sistema funciona correctamente, sino también los casos en que falla, los contextos en que puede ser malinterpretado y los actores que pueden verse afectados por sus decisiones.

La tecnología funciona como un complemento al trabajo humano, no como su reemplazo. Mantener esa distinción clara —en el diseño, en la interfaz y en la comunicación con los usuarios— es parte integral de un desarrollo de inteligencia artificial ético y responsable.
