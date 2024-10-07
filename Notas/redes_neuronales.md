# Introducción a las Redes Neuronales con Python y TensorFlow
Las redes neuronales son modelos computacionales inspirados en el funcionamiento del cerebro humano, diseñadas para reconocer patrones y realizar predicciones. Estos modelos están compuestos por neuronas, organizadas en capas, conectadas entre sí mediante pesos y sesgos. A través del entrenamiento, las redes neuronales ajustan estos valores para mejorar su precisión en la resolución de problemas. En este texto, te guiaré paso a paso en la creación de una red neuronal sencilla usando Python y TensorFlow en Visual Studio Code.

### Conceptos Básicos
**1. Neurona Artificial**


Una neurona artificial es la unidad fundamental de una red neuronal. Similar a una neurona biológica, recibe múltiples entradas, procesa la información y produce una salida. Matemáticamente, una neurona realiza una operación de suma ponderada de las entradas, añade un sesgo (bias) y aplica una función de activación para determinar la salida.

```

Fórmula de una neurona: 

Salida = Función de Activación (∑ (Peso * Entradas) + Sesgo)

```

**2. Peso**


Los pesos son valores que determinan la importancia de las entradas. Cada conexión entre neuronas tiene un peso que se ajusta durante el entrenamiento para optimizar los resultados.

**3. Sesgo**


El sesgo es un valor que se suma a la combinación ponderada de las entradas de una neurona. Permite que la red neuronal ajuste su salida incluso cuando todas las entradas sean cero.

**4. Capa (Layer)**


Una capa es un conjunto de neuronas que trabajan en paralelo. Existen diferentes tipos de capas, como:

- Capa de Entrada: Recibe los datos iniciales.
- Capa Oculta: Procesa la información entre la capa de entrada y la de salida.
- Capa de Salida: Produce el resultado final de la red.

**5. Diagrama de una Red Neuronal**


Visualmente, una red neuronal se representa como un grafo donde:

- Nodos: Representan las neuronas.
- Conexiones: Representan los pesos entre neuronas.
- Capas: Agrupan las neuronas según su función en la red.

Figura: Diagrama esquemático de una red neuronal simple.

**6. Entrenamiento de una Red Neuronal**


El entrenamiento es el proceso mediante el cual la red ajusta sus pesos y sesgos para aprender de los datos. Este proceso consta de varios pasos:

1. Forward Pass (Propagación hacia adelante): La red procesa las entradas a través de las capas para producir una salida.


2. Cálculo de la Pérdida (Loss): Se evalúa el error entre la salida predicha y el valor real utilizando una función de pérdida, como el error cuadrático medio.


3. Backward Pass (Propagación hacia atrás): Se calcula el gradiente de la pérdida respecto a cada peso y sesgo mediante el algoritmo de retropropagación.


4. Actualización de Pesos: Se ajustan los pesos y sesgos en la dirección que minimiza la pérdida, utilizando un optimizador como Adam o SGD.


Este proceso se repite en múltiples iteraciones llamadas épocas hasta que la red alcanza un nivel aceptable de precisión.

## Creando una Red Neuronal con Python y TensorFlow

A continuación, desarrollaremos una red neuronal simple que convierte temperaturas de Celsius a Fahrenheit. Utilizaremos TensorFlow, una de las bibliotecas más populares para el aprendizaje profundo, y Visual Studio Code como entorno de desarrollo.

**Paso 1: Instalación de Dependencias**


Antes de comenzar, asegúrate de tener Python instalado. Luego, instala TensorFlow y NumPy utilizando pip:
```
pip install tensorflow numpy
```

**Paso 2: Configuración del Entorno en Visual Studio Code**


* Abre Visual Studio Code.
* Crea un nuevo archivo llamado red_neuronal.py.
* Importa las bibliotecas necesarias:
```
import tensorflow as tf
import numpy as np
```

**Paso 3: Preparación de los Datos**

Definiremos los datos de entrada (temperaturas en Celsius) y las etiquetas (temperaturas en Fahrenheit).

```
# Datos de entrada (Celsius) y etiquetas (Fahrenheit)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100], dtype=float)
```
**Paso 4: Construcción del Modelo** 

Crearemos una red neuronal secuencial con una sola capa densa (fully connected).

```
# Definición de una capa densa con 1 unidad (neurona)
capa = tf.keras.layers.Dense(units=1, input_shape=[1], activation='linear')

# Construcción del modelo secuencial
modelo = tf.keras.Sequential([capa])
```

### Explicación:

* **units=1:** La capa tiene una sola neurona.
* **input_shape=[1]:** La entrada tiene una dimensión de 1 (temperatura en Celsius).
* **activation='linear':** La función de activación es lineal, adecuada para problemas de regresión.

**Paso 5: Compilación del Modelo**

Configuramos el modelo con un optimizador y una función de pérdida.

```
# Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
```
### Detalles:

* **optimizer=tf.keras.optimizers.Adam(0.1):** Utilizamos el optimizador Adam con una tasa de aprendizaje de 0.1.
* **loss='mean_squared_error':** La función de pérdida es el error cuadrático medio, común en problemas de regresión.

**Paso 6: Entrenamiento del Modelo**

Entrenamos la red neuronal utilizando los datos proporcionados.

```
# Entrenamiento del modelo
modelo.fit(celsius, fahrenheit, epochs=500)
```
### Parámetros:

* **celsius:** Datos de entrada.
* **fahrenheit:** Etiquetas.
* **epochs=500:** Número de iteraciones de entrenamiento.

**Paso 7: Evaluación y Uso del Modelo**

Después de entrenar, podemos usar el modelo para hacer predicciones.

```
# Predicción
celsius_nuevo = 100
prediccion = modelo.predict([celsius_nuevo])
print(f"{celsius_nuevo}°C son aproximadamente {prediccion[0][0]:.2f}°F")
```
### Explicación del Proceso de Entrenamiento
1. Inicialización: La red comienza con pesos y sesgos aleatorios.
2. Forward Pass: Las entradas (Celsius) se pasan a través de la red para obtener las salidas predichas (Fahrenheit).
3. Cálculo de la Pérdida: Se compara la salida predicha con la salida real utilizando la función de pérdida (error cuadrático medio).
4. Backward Pass: Se calcula el gradiente de la pérdida respecto a cada peso y sesgo.
5. Actualización de Pesos: Los pesos y sesgos se ajustan en la dirección que reduce la pérdida, utilizando el optimizador Adam.
6. Repetición: Este proceso se repite durante 500 épocas hasta que la red aprende a realizar la conversión con precisión.