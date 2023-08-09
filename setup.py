import os
import subprocess
import urllib.request
import tarfile

def main():
    repo_url = "https://github.com/serg6io4/ChessProject/archive/refs/heads/main.tar.gz"
    project_folder = "ChessProject-main"  # Nombre de la carpeta del proyecto una vez extraído

    # Descarga y extrae el repositorio de GitHub
    urllib.request.urlretrieve(repo_url, "project.tar.gz")
    with tarfile.open("project.tar.gz", "r:gz") as tar:
        tar.extractall()

    # Cambia al directorio del proyecto
    os.chdir(project_folder)

    # Instala las dependencias
    subprocess.call(["pip", "install", "-r", "requirements.txt"])

    print("Proyecto descargado y configurado.")

if __name__ == "__main__":
    main()
