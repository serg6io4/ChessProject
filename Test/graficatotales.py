import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carga los datos desde el archivo Excel
excel_file = "Test\Testeo3.xlsx"
data = pd.read_excel(excel_file)

# Calcula la media y desviación estándar de la tasa de acierto por plataforma
tasa_acierto_por_plataforma = data.groupby('Plataforma')['Tasa_acierto']
media_tasa_acierto = tasa_acierto_por_plataforma.mean()
desviacion_tasa_acierto = tasa_acierto_por_plataforma.std()

# Crea la gráfica de barras con media y desviación estándar
plt.figure(figsize=(10, 6))
ax = media_tasa_acierto.plot(kind='bar', yerr=desviacion_tasa_acierto, capsize=4, color='skyblue')
plt.xlabel('Plataforma')
plt.ylabel('Tasa de Acierto')
plt.title('Media y Desviación Típica de Tasa de Acierto por Plataforma')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Añadir más números entre 0 y 1 en el eje y (vertical)
extra_ticks = np.linspace(0, 1, 11)
plt.yticks(list(plt.yticks()[0]) + list(extra_ticks))

# Limitar el rango del eje y entre 0 y 1
plt.ylim(0, 1)

# Muestra la gráfica
plt.show()
