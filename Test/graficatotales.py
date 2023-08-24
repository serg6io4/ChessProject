import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carga los datos desde el archivo Excel
excel_file = "Test\Testeo.xlsx"
data = pd.read_excel(excel_file)

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

# Crear la segunda gráfica: Media y Desviación Típica de Tasa de Acierto por Plataforma y Tipo
tasa_acierto_por_plataforma_tipo = data.groupby(['Plataforma', 'R/S'])['Tasa_acierto']
media_tasa_acierto_tipo = tasa_acierto_por_plataforma_tipo.mean()
desviacion_tasa_acierto_tipo = tasa_acierto_por_plataforma_tipo.std()

plt.figure(figsize=(8, 6))
ax = media_tasa_acierto_tipo.unstack().plot(kind='bar', yerr=desviacion_tasa_acierto_tipo.unstack(), capsize=4)
plt.xlabel('Plataforma')
plt.ylabel('Tasa de Acierto')
plt.title('Media y Desviación Típica de Tasa de Acierto por Plataforma y Tipo')
plt.xticks(rotation=45)
plt.legend(['Sintética', 'Real'])
plt.grid(True)
plt.tight_layout()
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11))
plt.show()
