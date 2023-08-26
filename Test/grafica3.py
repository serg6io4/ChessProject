import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo Excel en un DataFrame
excel_file = 'Test\Testeo.xlsx'  # Cambia esto a la ubicación real del archivo
df = pd.read_excel(excel_file)

# Calcular la media de los atributos de color
df['Color'] = df[['ColorR', 'ColorG', 'ColorB']].mean(axis=1)

# Atributos
atributos = ['Gamma', 'Brillo', 'Sigma', 'Color']

# Iterar sobre los atributos y crear las gráficas de líneas por separado
for atributo in atributos:
    plt.figure(figsize=(10, 6))

    grouped = df.groupby(atributo)['Tasa_acierto'].agg(['mean', 'std'])
    x = grouped.index
    y_mean = grouped['mean']
    y_std = grouped['std']

    plt.errorbar(x, y_mean, yerr=y_std, linestyle='-', marker='o', capsize=5)
    plt.title(f'Media de Tasa de Acierto vs {atributo}')
    plt.xlabel(atributo)
    plt.ylabel('Media de Tasa de Acierto')
    plt.tight_layout()
    plt.show()
