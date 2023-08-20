import pandas as pd
import matplotlib.pyplot as plt

# Carga los datos desde el archivo Excel
excel_file = "Test\Testeo3.xlsx"
data = pd.read_excel(excel_file)

# Calcula la media y desviación estándar de la tasa de acierto por plataforma
tasa_acierto_por_plataforma = data.groupby('Plataforma')['Tasa_acierto']
media_tasa_acierto = tasa_acierto_por_plataforma.mean()
desviacion_tasa_acierto = tasa_acierto_por_plataforma.std()

# Calcula la media y desviación estándar de la tasa de acierto por plataforma y r/s
tasa_acierto_por_plataforma_rs = data.groupby(['Plataforma', 'R/S'])['Tasa_acierto']
media_tasa_acierto_rs = tasa_acierto_por_plataforma_rs.mean().unstack()
desviacion_tasa_acierto_rs = tasa_acierto_por_plataforma_rs.std().unstack()

# Atributos a considerar (sin colorR, colorG, colorB)
atributos = ['Gamma', 'Brillo', 'ColorR', 'ColorG', 'ColorB', 'Sigma']

# Agrupa los atributos de colorR, colorG y colorB en uno solo llamado Color y calcula la media
data['Color'] = data[['ColorR', 'ColorG', 'ColorB']].mean(axis=1)
atributos.remove('ColorR')
atributos.remove('ColorG')
atributos.remove('ColorB')

# Cantidad de intervalos en el eje x
num_intervals = 10

# Calcula las medias y desviaciones de la tasa de acierto por atributo en las muestras sintéticas
media_acierto_por_atributo = {}
desviacion_acierto_por_atributo = {}
for atributo in atributos:
    # Excluir filas con valores nulos (None) en el atributo actual
    filtered_data = data[data[atributo].notnull()]
    interval_data = pd.cut(filtered_data[atributo], bins=num_intervals)
    data_grouped = filtered_data.groupby(interval_data)['Tasa_acierto']
    media_acierto_por_atributo[atributo] = data_grouped.mean()
    desviacion_acierto_por_atributo[atributo] = data_grouped.std()

# Excluir filas con valores nulos (None) en el atributo "Sigma"
filtered_data_sigma = data[data['Sigma'].notnull()]
min_sigma_value = max(0, filtered_data_sigma['Sigma'].min())  # Asegurar valor mínimo de 0
filtered_data_sigma = filtered_data_sigma[filtered_data_sigma['Sigma'] >= min_sigma_value]
interval_data_sigma = pd.cut(filtered_data_sigma['Sigma'], bins=num_intervals, right=False)  # Evitar incluir el límite derecho
data_grouped_sigma = filtered_data_sigma.groupby(interval_data_sigma)['Tasa_acierto']
media_acierto_por_atributo['Sigma'] = data_grouped_sigma.mean()
desviacion_acierto_por_atributo['Sigma'] = data_grouped_sigma.std()

# Gráfica para el atributo Color con intervalos en el eje x
plt.figure(figsize=(10, 6))
interval_data_color = pd.cut(data['Color'], bins=num_intervals)
media_acierto_por_atributo['Color'] = data.groupby(interval_data_color)['Tasa_acierto'].mean()
desviacion_acierto_por_atributo['Color'] = data.groupby(interval_data_color)['Tasa_acierto'].std()
media_acierto_por_atributo['Color'].plot(style='.-', yerr=desviacion_acierto_por_atributo['Color'], capsize=4)
plt.xlabel('Color')
plt.ylabel('Tasa de Acierto')
plt.title('Media y Desviación Típica de Tasa de Acierto por Color')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Crea las gráficas restantes con líneas en lugar de barras
line_styles = ['-','--','-.',':']
for i, atributo in enumerate(['Gamma', 'Sigma', 'Brillo']):
    plt.figure(figsize=(10, 6))
    media_acierto_por_atributo[atributo].plot(style=line_styles[i], yerr=desviacion_acierto_por_atributo[atributo], capsize=4)
    plt.xlabel(atributo)
    plt.ylabel('Tasa de Acierto')
    plt.title(f'Media y Desviación Típica de Tasa de Acierto por {atributo}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

# Muestra todas las gráficas
plt.show()
