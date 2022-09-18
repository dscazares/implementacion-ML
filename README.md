# Implementación de técnicas de aprendizaje máquina

En este repositorio se pueden encontrar 2 implementaciones de técnicas de machine learning, una desde cero y otra usando un framework, así como un análisis del desempeño de uno de los modelos

**Tabla de contenidos**
<!-- no toc -->
1. [Dataset](#dataset-utilizado)
2. [Implementación "from scratch"](#implementación-from-scratch)
3. [Implementación con framework](#implementación-con-uso-de-un-framework)
4. [Análisis del desempeño](#análisis-y-reporte-del-desempeño-de-un-modelo)


## Dataset utilizado

Para ambas implementaciones se utilizo el dataset [Real estate price prediction](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set) (`Real estate.csv`) obtenido de un repositorio de University of California Irvine, el cual consiste en datos históricos del valor de bienes raíces en Taipei, Taiwán.

**Objetivo**: Predecir el valor de Y house price of unit area con base en las variables X.

**Variables**
1. **X1 transaction date**: Fecha de la transacción. Formato: año.mes_como_% (Ej. Marzo 2013 = 2013.250) 
2. **X2 house age**: Antigüedad de la propiedad en años.
3. **X3 distance to the nearest MRT station**: Distancia en metros a la estación de tren (Mass Rapid Transit / MRT) más cercana.
4. **X4 number of convenience stores**: Número de tiendas de conveniencia cercanas
5. **X5 latitude**: Latitud (Coordenada geográfica)
6. **X6 longitude**: Longitud (Coordenada geográfica)
7. **Y house price of unit area**: Precio de la propiedad en dolares taiwaneses por unidad de área (La unidad de area utilizada es 1 Ping, equivalente a 3.3 metros cuadrados)

## Implementación "from scratch"

En esta implementación se busco programar un algoritmo de aprendizaje máquina sin usar frameworks de machine learning o estadística avanzada.

#### Descripción

Se programo un algoritmo de **regresión lineal multiple** por medio de **gradiente descendiente**.

Lo que se hace con este algoritmo es buscar los mejores coeficientes para la fórmula de la ecuación lineal a través de un proceso de optimización iterativo que utiliza las derivadas de los  coeficientes para minimizar el error.

#### Implementación

Se tienen 2 archivos: uno para el modelo y otro para la ejecución.  

En `model.py` se implementa una clase que incluye las funciones `fit()` y `predict()` para ajustar el modelo (con gradiente descendiente) y obtener predicciones. Así mismo, se incluyen las funciones auxiliares `data_split()`, `mse()` y `rmse()` para dividir los datos en conjuntos de entrenamiento y prueba, así como calcular el error de la predicción.

En `from_scratch.py` se realizan los pasos necesarios para utilizar el modelo programado: lectura de datos, división en conjuntos, escalamiento de variables de entrenamiento, ajuste, predicción y cálculo del error.

#### Ejecución y resultados

Para usar el modelo es necesario descargar los archivos `model.py` y `from_scratch.py`, y ejectar este útlimo.

Al ejecutarlo se obtiene la ecuación de regresión multiple calculada, así como el error que existe entre los datos de prueba y los predichos.

```
Equation: y =  37.89607250755148 + -3.207089376786645 x1 + -5.089159686724484 x2 + 3.491601488746529 x3 + 3.2751004128453967 x4 + -0.11701553683897832 x5
MSE:  61.58165532934254
RMSE:  7.847397997383753
```

## Implementación con uso de un framework

En esta implementación se busco programar un algoritmo de aprendizaje máquina por medio del uso de un  framework.

#### Descripción

Se programo un algoritmo de **Random Forest Regressor** con la librería **sklearn**.

#### Implementación

En `with_framework.py` se realizan los pasos necesarios para utilizar el modelo: lectura de datos, división en conjuntos, escalamiento de variables de entrenamiento, ajuste, predicción y cálculo del error.

#### Ejecución y resultados

Para usar el modelo es necesario descargar el archivo `with_framework.py` y ejectarlo.

Al ejecutarlo se obtiene el error que existe entre los datos de prueba y los predichos.

```
MSE: 52.060125360431684
RMSE: 7.215270290185371
```

## Análisis y reporte del desempeño de un modelo

Se escogio la implementación de **Random Forest Regressor con sklearn** para analizar su desempeño antes y después de utilizar técnicas de regularización y ajuste de parámetros para mejorarlo.