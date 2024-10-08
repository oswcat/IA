import tensorflow as tf
import numpy as np
# anadir 10 valores como tarea de celcius y far de 20 grados cada uno y redondear desde .6 grados
celcius  = np.array([-40, -10 , 0, 8, 15, 22,38], dtype=float)
fahrenheit =  np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo  = tf.keras.models.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error')

print("Comenzando el entrenamiento...")
historial = modelo.fit(celcius,  fahrenheit, epochs=500, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("# Magnitud de perdidas")
plt.plot(historial.history['loss'])
plt.show()

print("Hagamos una prediccion.")
resultado = modelo.predict ([100])
print("El resultado es :" + str(resultado ) + "fahrenheit")

