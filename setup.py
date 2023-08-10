import os
import argparse
import subprocess
from git import Repo

def main(args):
    repo_url = "https://github.com/serg6io4/ChessProject.git"
    project_folder = args.ruta

    # Clona el repositorio de GitHub
    if not os.path.exists(project_folder):
        Repo.clone_from(repo_url, project_folder)
    else:
        print(f"El directorio '{project_folder}' ya existe.")

    # Cambia al directorio del proyecto
    os.chdir(project_folder)

    # Instala las dependencias
    subprocess.call(["pip", "install", "-r", "requirements.txt"])

    print("Proyecto descargado y configurado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clona un repositorio de GitHub y configura el proyecto.")
    parser.add_argument("ruta", help="Ruta donde deseas clonar el repositorio y configurar el proyecto.")
    args = parser.parse_args()
    
    main(args)

