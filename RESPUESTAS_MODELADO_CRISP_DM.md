# Fase de Modelado (CRISP-DM) - Caso Banco Berka
## Respuestas a las Preguntas de Negocio y Modelado Predictivo

---

### Pregunta 1: ¿Qué tipo de problema se está resolviendo y qué técnica de modelado es la más adecuada?

#### 1. Tipo de Problema
Se está resolviendo un problema de **Clasificación Binaria Supervisada** con **clases fuertemente desbalanceadas**.

* **Definición de la Variable Objetivo ($Y$ / `target`):**
  A partir de la información de préstamos del Banco Berka (`loan.asc`), la variable objetivo se define de forma dicotómica:
  * **$Y = 1$ (Buen pagador / Crédito sano):** Clientes con estatus `'A'` (préstamo liquidado con éxito) y estatus `'C'` (préstamo en curso al corriente de pagos). En la tabla minable depurada representa el **88,9 %** de los casos (601 clientes).
  * **$Y = 0$ (Mal pagador / Mora / Default):** Clientes con estatus `'B'` (préstamo no pagado / quebrado) y estatus `'D'` (préstamo en curso con deuda o atrasos graves). Representa el **11,1 %** de los casos (75 clientes).

* **Naturaleza del Negocio (Asimetría de Costos):**
  En la industria crediticia y bancaria, los errores de predicción no tienen el mismo costo:
  * **Falso Positivo:** Rechazar erróneamente a un cliente que habría pagado bien (costo de oportunidad: interés no ganado).
  * **Falso Negativo:** Otorgar el crédito a un cliente que caerá en mora (pérdida del capital prestado). Este costo es notablemente mayor.

#### 2. Técnica de Modelado más Adecuada
La técnica más adecuada para este problema son los **modelos lineales probabilísticos con regularización y ponderación de clases (`class_weight='balanced'`)**, específicamente la **Regresión Logística**, junto con modelos de ensamble de árboles (*Random Forest*) como comparación no lineal.

**Razones técnicas y regulatorias:**
1. **Estimación directa de probabilidades:** Modela la probabilidad condicional de cumplimiento $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$, permitiendo generar un puntaje (*Credit Scorecard*) y calibrar el punto de corte según el apetito de riesgo del banco.
2. **Estabilidad muestral:** Con un tamaño de muestra de $N = 676$ observaciones analíticas, los modelos lineales son más robustos frente al sobreajuste que los modelos complejos de caja negra.
3. **Explicabilidad y Cumplimiento Normativo (Basilea II/III):** Los reguladores financieros exigen que las entidades bancarias puedan auditar y justificar con precisión los factores por los cuales se concede o deniega un financiamiento.

---

### Pregunta 2: ¿Cómo se dividirán y validarán los datos para probar el modelo de manera justa?

Para asegurar una evaluación imparcial, reproducible y sin fuga de información (*data leakage*), se implementa una estrategia en dos niveles:

```
[ Tabla Minable Total: 676 registros ]
           │
           ├────────────────────────────┬─────────────────────────────┐
           ▼                            ▼                             ▼
   Exclusión de Llave           Conjunto Train (80%)          Conjunto Test (20%)
   `client_id` (no entra)        540 observaciones             136 observaciones
                                (480 Buenos / 60 Morosos)     (121 Buenos / 15 Morosos)
                                        │                             │
                                        ▼                             ▼
                               Stratified 5-Fold CV           Evaluación Final Ciega
                               (Ajuste y estabilidad)         (Desempeño real en negocio)
```

1. **Partición Holdout Train / Test (80 % / 20 %):**
   * **Entrenamiento (`X_train`, `y_train`):** 540 observaciones.
   * **Prueba (`X_test`, `y_test`):** 136 observaciones.
   * **Estratificación obligatoria (`stratify=y`):** Es imprescindible aplicar estratificación para garantizar que tanto el conjunto de entrenamiento como el de prueba mantengan con exactitud la tasa de impago original (11,1 % de morosos: 60 en Train y 15 en Test).
   * **Exclusión de identificadores:** La variable `client_id` se remueve del espacio de características predictoras para evitar que el algoritmo memorice índices de clientes.

2. **Validación Cruzada Estratificada (*Stratified 5-Fold Cross-Validation*):**
   * Se ejecuta **exclusivamente dentro del 80 % de entrenamiento**, sin tocar el conjunto de prueba.
   * Divide los datos en 5 particiones preservando la proporción 89/11 en cada pliegue. Permite estimar la media y la desviación típica de las métricas para verificar que el modelo es estable y no depende de una partición afortunada.

3. **Conjunto de Prueba Ciego (Holdout Test Set):**
   * El 20 % restante se utiliza una única vez al final del experimento como estimador no sesgado del rendimiento del modelo frente a nuevas solicitudes de crédito en producción.

---

### Pregunta 3: ¿Qué modelos candidatos se van a entrenar y comparar?

Se seleccionó un portafolio de 6 modelos con distintos sesgos inductivos y capacidades de manejo del desbalance de clases:

| Modelo Candidato | Familia / Algoritmo | Mecanismo ante Desbalance | Propósito en el Experimento |
|---|---|---|---|
| **1. Dummy Classifier** | Modelo Nulo (Heurística) | Predice siempre la clase mayoritaria ($Y=1$) | Establece la cota mínima de comparación (*baseline*). |
| **2. Regresión Logística (Balanced)** | Paramétrico Lineal | Penalización inversa de frecuencias (`class_weight='balanced'`) | Estándar de la industria bancaria; máxima interpretabilidad. |
| **3. Árbol de Decisión CART** | No paramétrico (Reglas) | Ponderación balanceada (`class_weight='balanced'`) | Reglas de corte jerárquicas; podado a profundidad 4 para evitar sobreajuste. |
| **4. Random Forest** | Ensamble Bagging | Muestreo estratificado con pesos balanceados | Reduce la varianza promediando 150 árboles decorrelacionados. |
| **5. Gradient Boosting Classifier** | Ensamble Boosting | Optimización secuencial sobre residuos | Evalúa si el aprendizaje por gradiente mejora la discriminación. |
| **6. HistGradientBoosting (Balanced)** | Boosting por Histogramas | Ponderación de pesos por clase | Implementación eficiente de boosting adaptada a desbalance. |

---

### Pregunta 4: ¿Qué métricas se usarán para evaluar el desempeño del modelo?

> **Advertencia sobre la Exactitud (*Accuracy*):**
> En un dataset donde el 88,9 % de los clientes son cumplidores, un modelo ingenuo que apruebe todos los préstamos obtiene un **Accuracy del 88,9 %**, pero su tasa de detección de morosos es del **0 %**, lo que llevaría al banco a la insolvencia. Por ello, el Accuracy se descarta como criterio principal.

Las métricas bancarias adoptadas son:

1. **Estadístico KS (Kolmogorov-Smirnov):**
   Es la métrica estándar de la banca mundial para medir la capacidad de discriminación crediticia. Mide la distancia máxima absoluta entre la distribución acumulada de buenos pagadores y la distribución acumulada de morosos:
   $$KS = \max_{s} |F_{\text{buenos}}(s) - F_{\text{malos}}(s)|$$
   * **Criterio bancario:**
     * $KS < 20\,\%$ : Capacidad discriminatoria insuficiente.
     * $20\,\% \le KS \le 40\,\%$ : **Rango estándar / operativo aceptado en banca**.
     * $KS > 40\,\%$ : Excelente capacidad de separación de riesgo.

2. **ROC-AUC (Área bajo la Curva ROC):**
   Mide la probabilidad de que el modelo asigne un puntaje de riesgo mayor a un cliente en mora que a un buen pagador seleccionado al azar, evaluando todos los posibles puntos de corte de forma global.

3. **Recall de la Clase Mora ($Y = 0$):**
   $$\text{Recall}_{\text{Mora}} = \frac{\text{Morosos Detectados (Verdaderos Negativos)}}{\text{Total de Morosos Reales}}$$
   Mide qué porcentaje de los créditos que terminarán en impago el banco es capaz de interceptar a tiempo para no sufrir pérdidas.

4. **Precision de la Clase Mora ($Y = 0$) y F1-Macro:**
   Evalúan el costo de balance para que el banco no rechace masivamente a buenos clientes para atrapar a los morosos.

---

### Pregunta 5: ¿Cuál modelo se selecciona como mejor y por qué?

#### Resultados Empíricos Comparativos

Tras ejecutar el experimento en [MODELADO_BERKA.ipynb](file:///c:/Users/USUARIO/Desktop/proy/PRACTICA_1-MODULO-3/MODELADO_BERKA.ipynb) sobre los 676 registros de la tabla minable:

| Modelo Candidato | CV ROC-AUC | CV KS (%) | CV Recall Mora (0) | Test ROC-AUC | Test KS (%) | Test Recall Mora (0) | Test F1-Macro | Test Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regresión Logística (Balanced)** | **0,6424 ± 0,075** | **31,67 %** | **0,5000** | **0,6094** | **32,12 %** | **0,6000** | **0,5098** | 63,24 % |
| **Random Forest (Balanced)** | 0,6462 ± 0,083 | 33,33 % | 0,3000 | 0,5603 | 22,92 % | 0,1333 | 0,4614 | 71,32 % |
| **Árbol de Decisión (Balanced)** | 0,6036 ± 0,113 | 25,00 % | 0,6833 | 0,5350 | 9,64 % | 0,5333 | 0,4191 | 50,74 % |
| **Gradient Boosting** | 0,6411 ± 0,118 | 37,92 % | 0,0667 | 0,5267 | 30,52 % | 0,0000 | 0,4688 | 88,24 % |
| **HistGradientBoosting (Balanced)** | 0,6047 ± 0,089 | 35,42 % | 0,1000 | 0,5069 | 27,27 % | 0,0000 | 0,4603 | 85,29 % |
| **Dummy (Baseline)** | 0,5000 ± 0,000 | 0,00 % | 0,0000 | 0,5000 | 0,00 % | 0,0000 | 0,4708 | 88,97 % |

---

#### Modelo Seleccionado
Se selecciona como **mejor modelo** a la **Regresión Logística con Ponderación Balanceada (`class_weight='balanced'`)**.

#### Justificación de la Elección:
1. **Mayor Capacidad de Detección de Impagos (Recall = 60,0 % en Test):**
   En el conjunto de prueba independiente (136 clientes, 15 de ellos morosos), la Regresión Logística detectó y frenó a **9 de los 15 morosos**. En contraste:
   * Los modelos de Boosting (*Gradient Boosting* y *HistGradientBoosting*) tuvieron un **Recall de 0,0 %** (dejaron pasar al 100 % de los créditos fallidos debido al desbalance).
   * *Random Forest* solo detectó a 2 de los 15 morosos (Recall de 13,3 %).

2. **Estadístico KS Óptimo y Robusto (32,12 % en Test):**
   Mantiene una separación de distribuciones superior al 30 % tanto en Validación Cruzada (31,67 %) como en el conjunto de prueba (32,12 %), cumpliendo plenamente el estándar operativo de riesgo crediticio bancario.

3. **Excelente Generalización sin Sobreajuste:**
   Mientras los modelos no lineales basados en árboles sufrieron caídas drásticas al ser evaluados en datos ciegos (el ROC-AUC de Random Forest cayó de 0,646 en CV a 0,560 en Test, y el KS del árbol cayó a 9,6 %), la Regresión Logística demostró una consistencia notable (0,642 en CV vs 0,609 en Test).

4. **Interpretabilidad Financiera de los Coeficientes:**
   Los coeficientes estimados por el modelo coinciden con los fundamentos de la teoría financiera:
   * **`amount_z` (Coef: -0,7525):** Es el factor de riesgo más determinante. Cuanto mayor es el capital prestado, mayor es la probabilidad de que el cliente incumpla.
   * **`duration_z` (Coef: +0,3015):** A mayor plazo de amortización, menor es el impacto de la cuota mensual en los ingresos del cliente, reduciendo el riesgo de mora.
   * **`frecuencia_por_transaccion` (Coef: -0,7137):** Los clientes que reciben estados de cuenta tras cada transacción presentan mayor volatilidad y riesgo que quienes operan con extracto mensual consolidado (`frecuencia_mensual`: +0,3867).
   * **Efecto geográfico:** Se identificaron diferencias regionales consistentes con la economía checa de la época (la región de *North Bohemia* presenta menor morosidad en relación con *North Moravia*).

