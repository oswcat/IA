import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler   

# Datos
tiempo = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495], dtype=float)
trafico = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310], dtype=float)

# Normalización de datos
scaler = MinMaxScaler()
tiempo_scaled = scaler.fit_transform(tiempo.reshape(-1, 1))
trafico_scaled = scaler.fit_transform(trafico.reshape(-1, 1))

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(tiempo_scaled, trafico_scaled, test_size=0.2, random_state=42)

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compilar
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar
historial = modelo.fit(X_train, y_train, epochs=400, validation_data=(X_test, y_test))

# Evaluar
loss = modelo.evaluate(X_test, y_test)
print("Pérdida en el conjunto de prueba:", loss)

# Visualizar
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()   

# Predecir
nuevo_tiempo = np.array([[100.0]])  # Asegurarse de tener la misma forma que los datos de entrenamiento
nuevo_tiempo_scaled = scaler.transform(nuevo_tiempo)
prediccion_scaled = modelo.predict(nuevo_tiempo_scaled)
prediccion = scaler.inverse_transform(prediccion_scaled)[0][0]
print("La predicción es:", prediccion, "tráfico de red")