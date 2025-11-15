# Conclusiones
A lo largo del proceso de experimentación y análisis, se desarrolló un enfoque integral de clasificación para la predicción del riesgo de cáncer tiroideo, utilizando técnicas avanzadas de preprocesamiento, balanceo de datos, optimización de hiperparámetros (GridSearchCV) y evaluación con métricas centradas en el AUC y el recall multiclase, con el fin de garantizar una interpretación robusta del rendimiento de los modelos.

En las primeras etapas, el análisis exploratorio permitió identificar la alta correlación entre ciertas variables clínicas, así como la presencia de desbalance de clases en el conjunto de datos. Esto justificó la aplicación de técnicas como SMOTE, las cuales contribuyeron a mejorar la capacidad de los clasificadores para reconocer instancias minoritarias, especialmente en las clases Low y Medium, que tienden a ser subrepresentadas.

Los resultados de las distintas pruebas mostraron que el desempeño de los modelos varía significativamente según la arquitectura y los parámetros de entrenamiento. En general, los modelos basados en árboles de decisión y boosting (como Random Forest y XGBoost) obtuvieron los valores más altos de AUC macro, alcanzando cifras entre 0.73 y 0.75, con especial dominio en la clase High, cuyo AUC fue cercano a 1.000 en varios experimentos. Sin embargo, la precisión en las clases Low y Medium se mantuvo por debajo de 0.65, evidenciando la dificultad inherente del modelo para capturar patrones menos representativos.

El análisis explicativo mediante LIME y SHAP aportó una comprensión más profunda de la lógica interna de los modelos, revelando qué variables clínicas influyen más en la predicción. Por ejemplo, indicadores relacionados con el nivel hormonal, historial familiar y exposición a radiación emergieron como los factores más determinantes para la clase de alto riesgo (High), confirmando su relevancia médica.

Asimismo, el proceso iterativo de tuning de modelos demostró que pequeñas variaciones en los hiperparámetros pueden mejorar marginalmente el desempeño global, aunque la ganancia tiende a estabilizarse una vez que el modelo logra una representación adecuada del espacio de características. Este hallazgo refuerza la importancia de combinar una buena selección de atributos con estrategias de interpretación y evaluación multicriterio.

En síntesis, el estudio logró establecer una línea base sólida para la clasificación del riesgo de cáncer tiroideo, destacando que:

XGBoost ofrece el mejor equilibrio entre rendimiento y estabilidad;

Los modelos lineales (Logistic Regression) presentan un desempeño inferior, especialmente ante desbalance de clases;

La incorporación de técnicas explicativas (LIME, SHAP) es esencial para la transparencia y validación clínica de los resultados;

Y finalmente, el trabajo abre la puerta a la exploración de nuevos modelos —como CatBoost o TabNet— que podrían mejorar el recall multiclase, métrica prioritaria para reducir falsos negativos en contextos médicos.

# Investigacion en propuesta de machine learning

## Modelos derivados o mejorados de KNN, RandomForest y XGBoost (no usados en estudios de Thyroid Cancer)

Los siguientes enfoques fueron identificados en *Clasificacion.txt* y no aparecen en los estudios de cáncer de tiroides, lo que los hace relevantes y originales para incorporar en investigaciones futuras:

| **Modelo Base** | **Variante Mejorada Encontrada en Clasificacion.txt** | **¿Qué Mejora?** | **Potencial para el Proyecto** |
|------------------|-------------------------------------------------------|------------------|-------------------------------|
| **Random Forest (RF)** | Stacking/Voting Ensemble usando RF como base | Combina RF con SVM, ANN o Boosting → mejora la estabilidad en clasificación multiclase | Integrable en un *ensamble final* junto con los mejores modelos actuales |
| **Random Forest (RF)** | Feature Importance + RF optimizado con Grid/GA | Ajuste automático de hiperparámetros → aumento en Recall y precisión | Ideal para elevar el *Recall* sin cambiar de familia de modelos |
| **KNN** | KNN híbrido con selección de características (PCA/ReliefF) | Reduce ruido y mejora precisión en clases minoritarias | Útil para datasets con alta dimensionalidad |
| **XGBoost** | XGBoost combinado con técnicas de balanceo (SMOTE/ADASYN) | Combate el desbalance y mejora el F1-Score en problemas multiclase | Aplicable cuando existen clases poco representadas |
| **XGBoost + AutoML** | Integrado en sistemas AutoML/Multi-task | Auto-tuning de hiperparámetros sin intervención manual | Permite comparar tu XGBoost manual con versiones automatizadas |

---

### Técnicas no presentes en *ThyroidCancer.txt*, 100 % reutilizables

- **Stacked Ensembles** con RF, KNN o XGBoost  
- **Optimización** mediante Algoritmos Genéticos o Grid Search avanzado  
- **Híbridos con SMOTE/ADASYN** orientados a mejorar *Recall*  
- **Reducción de dimensionalidad previa a KNN** (PCA, ReliefF, LDA)

---

### Relevancia para el proyecto

| **Si ya se usó…** | **Posible innovación** |
|--------------------|------------------------|
| KNN base | KNN + selección de características o distancia adaptativa (MKNN) |
| RandomForest básico | RF + Stacking o RF + optimización evolutiva |
| XGBoost tradicional | XGBoost + SMOTE/ADASYN o integración en meta-ensemble |


[enlace de la investigacion](https://www.sciencedirect.com/science/article/pii/S2215016125003553?via%3Dihub)