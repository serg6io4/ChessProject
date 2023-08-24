import pandas as pd
import matplotlib.pyplot as plt

# Carga los datos desde el archivo Excel
excel_file = "Test\Testeo.xlsx"
data = pd.read_excel(excel_file)

# Filtrar las filas que no tienen valores None
data = data.dropna()

# Atributos a considerar
atributos_color = ['ColorR', 'ColorG', 'ColorB']
atributos_no_color = ['Gamma', 'Brillo', 'Sigma']

# Calcular la media de los atributos de color
data['Color_mean'] = data[atributos_color].mean(axis=1)

# Cantidad de intervalos en el eje x
num_intervals = 10

# Crear gráficas para los atributos con líneas y barras de error
line_styles = ['-','--','-.',':','-','--']
all_atributos = atributos_no_color + ['Color_mean']
for atributo in all_atributos:
    plt.figure(figsize=(10, 6))
    
    # Excluir filas con valores nulos (None) en el atributo actual
    filtered_data = data[data[atributo].notnull()]
    interval_data = pd.cut(filtered_data[atributo], bins=num_intervals)
    
    # Obtener el valor numérico representativo del centro de cada intervalo
    interval_centers = interval_data.apply(lambda x: x.mid).values
    
    # Calcular las medias y desviaciones agrupadas por los intervalos
    data_grouped = filtered_data.groupby(interval_centers)['Tasa_acierto']
    media_acierto = data_grouped.mean()
    desviacion_acierto = data_grouped.std()
    
    # Crear la gráfica
    plt.errorbar(media_acierto.index, media_acierto, yerr=desviacion_acierto, capsize=4, label=atributo, linestyle=line_styles[all_atributos.index(atributo)])
    plt.xlabel(atributo)
    plt.ylabel('Tasa de Acierto')
    plt.title(f'Media y Desviación Típica de Tasa de Acierto por {atributo}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Mostrar la gráfica
    plt.show()
