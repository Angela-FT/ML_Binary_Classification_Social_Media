# ML_Binary_Classification_Tweets

## Project Overview
This project focuses on building a binary classification model to analyze sentiment in social media posts, specifically Tweets. The goal is to classify Tweets as either positive or negative using machine learning techniques. Three models were compared to determine the best-performing one: Logistic Regression, Random Forest, and LightGBM.

## Business Problem
For this task, our business problem will focus on understanding trends and public opinions on various topics based on social media posts. For that, we will use an annotated Tweets dataset (positive and negative) and will train a ML model. The model will need to identify overall public sentiment and trends in topics like politics, social issues, or consumer preferences and our key requirements for the best model will be:

 **-Good performance on both classes.**

 **-Efficiency in handling large datasets.**

 **-Quick inference times.**

## Dataset
The dataset used in this project is **Sentiment140** (https://www.kaggle.com/datasets/kazanova/sentiment140), a publicly available dataset from Kaggle. It consists of annotated Tweets labeled as either positive or negative. The dataset provides a valuable resource for training sentiment analysis models by leveraging real-world social media data.

## Solution Approach
The approach followed in this project includes:
- Data preprocessing (cleaning and transforming text data)
- Exploratory Data Analysis (EDA) to understand dataset characteristics
- Feature engineering, including text vectorization techniques
- Training and evaluation of multiple machine learning models
- Selecting the best model based on performance metrics
- Final model interpretation and conclusion

### Chosen Models
Three machine learning models were selected for evaluation based on their strengths and applicability to text classification:
- **Logistic Regression**: A simple and efficient baseline model known for its interpretability and strong performance on text classification tasks.
- **Random Forest**: A powerful ensemble method that improves prediction accuracy by combining multiple decision trees and reducing overfitting.
- **LightGBM**: A gradient boosting model optimized for speed and efficiency, making it well-suited for large datasets and achieving state-of-the-art performance.

The models were compared based on accuracy, F1-score, and inference speed, with LightGBM and Logistic Regression emerging as the best-performing models.

## Repository Structure
```
project_root/
│── src/
│   ├── data_sample/       # A sample of the dataset
│   ├── models/            # The two models with best performance
│   ├── notebook/          # Test notebooks used during development
│   ├── results_notebook/  # Final notebook containing all code and explanations
│   ├── utils/             # Python scripts with utility functions
```

## Conclusion
The final model provides a solution for analyzing social media sentiment efficiently. Based on performance comparisons, LightGBM and Logistic Regression were chosen as the optimal models due to their balance between accuracy and speed.

---
# ML_Binary_Classification_Tweets

## Resumen del Proyecto
Este proyecto se centra en la construcción de un modelo de clasificación binaria para analizar el sentimiento en publicaciones de redes sociales, específicamente en Tweets. El objetivo es clasificar los Tweets como positivos o negativos utilizando técnicas de aprendizaje automático. Se compararon tres modelos para determinar el de mejor rendimiento: Regresión Logística, Random Forest y LightGBM.

## Problema de Negocio
Para esta tarea, nuestro problema de negocio se enfocará en comprender tendencias y opiniones públicas sobre diversos temas a partir de publicaciones en redes sociales. Para ello, utilizaremos un conjunto de datos de Tweets etiquetados (positivos y negativos) y entrenaremos un modelo de aprendizaje automático. El modelo deberá identificar el sentimiento general del público y las tendencias en temas como política, problemas sociales o preferencias de los consumidores. Nuestros requisitos clave para el mejor modelo son:

 **- Buen rendimiento en ambas clases.**

 **- Eficiencia en el manejo de grandes volúmenes de datos.**

 **- Tiempos de inferencia rápidos.**

## Conjunto de Datos
El conjunto de datos utilizado en este proyecto es **Sentiment140** (https://www.kaggle.com/datasets/kazanova/sentiment140), un dataset público disponible en Kaggle. Consiste en Tweets anotados y etiquetados como positivos o negativos. Este conjunto de datos proporciona un recurso valioso para entrenar modelos de análisis de sentimiento basados en datos reales de redes sociales.

## Enfoque de Solución
El enfoque seguido en este proyecto incluye:
- Preprocesamiento de datos (limpieza y transformación de texto)
- Análisis exploratorio de datos (EDA) para comprender las características del dataset
- Ingeniería de características, incluyendo técnicas de vectorización de texto
- Entrenamiento y evaluación de múltiples modelos de aprendizaje automático
- Selección del mejor modelo basado en métricas de rendimiento
- Interpretación del modelo final y conclusiones

### Modelos Seleccionados
Se seleccionaron tres modelos de aprendizaje automático para la evaluación, en función de sus fortalezas y aplicabilidad a la clasificación de texto:
- **Regresión Logística**: Un modelo base simple y eficiente, conocido por su interpretabilidad y buen rendimiento en tareas de clasificación de texto.
- **Random Forest**: Un poderoso método de ensamblado que mejora la precisión de predicción combinando múltiples árboles de decisión y reduciendo el sobreajuste.
- **LightGBM**: Un modelo de gradient boosting optimizado para velocidad y eficiencia, adecuado para grandes conjuntos de datos y capaz de lograr un rendimiento de vanguardia.

Los modelos fueron comparados en función de precisión, F1 y velocidad de inferencia, con LightGBM y Regresión Logística destacándose como los mejores modelos.

## Estructura del Repositorio
```
project_root/
│── src/
│   ├── data_sample/       # Una muestra del dataset
│   ├── models/            # Los dos modelos con mejor rendimiento
│   ├── notebook/          # Notebooks de prueba utilizados durante el desarrollo
│   ├── results_notebook/  # Notebook final con todo el código y explicaciones
│   ├── utils/             # Scripts en Python con funciones auxiliares
```

## Conclusión
El modelo final proporciona una solución eficiente para el análisis de sentimiento en redes sociales. Basado en comparaciones de rendimiento, LightGBM y Regresión Logística fueron elegidos como los modelos óptimos debido a su equilibrio entre precisión y velocidad.

---


