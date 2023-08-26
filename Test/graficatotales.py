import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo Excel en un DataFrame
excel_file = 'Test\Testeo.xlsx'  # Cambia esto a la ubicación real del archivo
df = pd.read_excel(excel_file)

# Crear una figura y ejes para la gráfica de Tasa de Acierto por Plataforma y R/S
plt.figure(figsize=(12, 6))

# Ordenar los valores de R/S para tener "R" a la izquierda y "S" a la derecha
df['R/S'] = pd.Categorical(df['R/S'], categories=["R", "S"], ordered=True)

# Gráfica de Tasa de Acierto por Plataforma y R/S
grouped_rs_platform = df.groupby(['Plataforma', 'R/S'])['Tasa_acierto'].mean().unstack()
grouped_rs_std_platform = df.groupby(['Plataforma', 'R/S'])['Tasa_acierto'].std().unstack()

grouped_rs_platform.plot(kind='bar', yerr=grouped_rs_std_platform, capsize=5, width=0.4)

plt.title('Media de Tasa de Acierto y Desviación Estándar por Plataforma y R/S')
plt.xlabel('Plataforma')
plt.ylabel('Media de Tasa de Acierto')
plt.legend(title='R/S')
plt.tight_layout()
plt.show()

# Crear una figura y ejes para la gráfica de Tasa de Acierto por Plataforma
plt.figure(figsize=(10, 6))

grouped_platform = df.groupby('Plataforma')['Tasa_acierto'].mean()
grouped_std_platform = df.groupby('Plataforma')['Tasa_acierto'].std()

x_platform = grouped_platform.index
y_platform = grouped_platform.values
y_std_platform = grouped_std_platform.values

plt.bar(x_platform, y_platform, yerr=y_std_platform, capsize=5)

plt.title('Media de Tasa de Acierto y Desviación Estándar por Plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Media de Tasa de Acierto')
plt.tight_layout()
plt.show()

