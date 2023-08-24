import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carga los datos desde el archivo Excel
excel_file = "Test\Testeo.xlsx"
data = pd.read_excel(excel_file)

# Asegurarse de que los valores de Tasa_acierto y Tasa_acierto1 estén en el rango [0, 1]
data['Tasa_acierto'] = np.clip(data['Tasa_acierto'], 0, 1)
data['Tasa_acierto1'] = np.clip(data['Tasa_acierto1'], 0, 1)

# Calcula la media y desviación estándar de la tasa de acierto por plataforma
tasa_acierto_por_plataforma = data.groupby('Plataforma')['Tasa_acierto']
media_tasa_acierto = tasa_acierto_por_plataforma.mean()
desviacion_tasa_acierto = tasa_acierto_por_plataforma.std()

# Crear la primera gráfica: Media y Desviación Típica de Tasa de Acierto por Plataforma
plt.figure(figsize=(8, 6))
ax = media_tasa_acierto.plot(kind='bar', yerr=desviacion_tasa_acierto, capsize=4, color='skyblue')
plt.xlabel('Plataforma')
plt.ylabel('Tasa de Acierto')
plt.title('Media y Desviación Típica de Tasa de Acierto por Plataforma')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11))
plt.show()

# Calcula la media y desviación estándar de la tasa de acierto1 por plataforma
tasa_acierto1_por_plataforma = data.groupby('Plataforma')['Tasa_acierto1']
media_tasa_acierto1 = tasa_acierto1_por_plataforma.mean()
desviacion_tasa_acierto1 = tasa_acierto1_por_plataforma.std()

# Crear la segunda gráfica: Media y Desviación Típica de Tasa de Acierto1 por Plataforma
plt.figure(figsize=(8, 6))
ax = media_tasa_acierto1.plot(kind='bar', yerr=desviacion_tasa_acierto1, capsize=4, color='salmon')
plt.xlabel('Plataforma')
plt.ylabel('Tasa de Acierto1')
plt.title('Media y Desviación Típica de Tasa de Acierto1 por Plataforma')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11))
plt.show()

