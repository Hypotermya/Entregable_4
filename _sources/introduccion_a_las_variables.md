# Introduccion a los datos

El presente análisis se basa en un conjunto de datos tomado de la plataforma Kaggle([Fuente de los datos](https://www.kaggle.com/datasets/mzohaibzeeshan/thyroid-cancer-risk-dataset/data)), el cual contiene información detallada relacionada con factores de riesgo del cáncer de tiroides. En total, el dataset incluye 212,691 registros, que abarcan datos demográficos, antecedentes clínicos, estilo de vida y niveles hormonales clave, con el objetivo de evaluar la probabilidad de desarrollar esta enfermedad.

Este conjunto de datos es ideal para aplicar técnicas de machine learning orientadas a la predicción del riesgo de cáncer de tiroides, así como para explorar relaciones significativas entre diversos factores y la aparición de esta condición médica.

Descripción de las Variables
El dataset se compone de 17 columnas, que se describen a continuación:

1. Patient_ID (int): Identificador único de cada paciente.

2. Age (int): Edad del paciente.

3. Gender (categórica): Género del paciente (Masculino/Femenino).

4. Country (categórica): País de residencia.

5. Ethnicity (categórica): Origen étnico del paciente.

6. Family_History (categórica): Antecedentes familiares de cáncer de tiroides (Sí/No).

7. Radiation_Exposure (categórica): Historia de exposición a radiación (Sí/No).

8. Iodine_Deficiency (categórica): Presencia de deficiencia de yodo (Sí/No).

9. Smoking (categórica): Hábito de fumar (Sí/No).

10. Obesity (categórica): Condición de obesidad (Sí/No).

11. Diabetes (categórica): Presencia de diabetes (Sí/No).

12. TSH_Level (float): Nivel de hormona estimulante de la tiroides (TSH) en µIU/mL.

13. T3_Level (float): Nivel de triyodotironina (T3) en ng/dL.

14. T4_Level (float): Nivel de tiroxina (T4) en µg/dL.

15. Nodule_Size (float): Tamaño de los nódulos tiroideos en cm.

16. Thyroid_Cancer_Risk (categórica): Riesgo estimado de cáncer de tiroides (Bajo/Medio/Alto).

17. Diagnosis (categórica): Diagnóstico final (Benigno/Maligno).

Este conjunto de datos permite realizar un análisis exploratorio enfocado en entender cómo varía el riesgo de cáncer de tiroides en función de diversas características como la edad, el género, los antecedentes familiares, la exposición a radiación, la presencia de deficiencia de yodo, y los niveles hormonales clave (TSH, T3 y T4), entre otros factores clínicos y de estilo de vida.

El objetivo de este análisis es identificar patrones relevantes que puedan ser útiles para el desarrollo de modelos predictivos de riesgo, así como para apoyar decisiones informadas en el ámbito clínico y de salud pública, mediante la detección temprana de perfiles de riesgo y posibles asociaciones significativas entre los distintos factores involucrados.
