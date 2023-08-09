import os
import shutil
import subprocess
from git import Repo

def main():
    repo_url = "https://github.com/serg6io4/ChessProject.git"
    project_folder = "ChessProject"  # Cambia esto si deseas un nombre diferente para la carpeta del proyecto

    # Clona el repositorio de GitHub
    Repo.clone_from(repo_url, project_folder)

    # Cambia al directorio del proyecto
    os.chdir(project_folder)

    # Instala las dependencias
    subprocess.call(["pip", "install", "-r", "requirements.txt"])

    # Crea la estructura de directorios si no existe
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    print("Proyecto descargado y configurado.")

if __name__ == "__main__":
    main()
