import tensorflow as tf
import numpy as np
# anadir 10 valores como tarea de celcius y far de 20 grados cada uno y redondear desde .6 grados
celcius  = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195], dtype=float)
fahrenheit =  np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo  = tf.keras.models.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error')

print("Comenzando el entrenamiento...")
historial = modelo.fit(celcius,  fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("# Magnitud de perdidas")
plt.plot(historial.history['loss'])
plt.show()

print("Hagamos una prediccion.")
resultado = modelo.predict (np.array([[100]]))
print("El resultado es :" + str(resultado ) + "fahrenheit")

