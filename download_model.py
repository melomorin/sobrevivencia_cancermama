import os
import gdown

# El ID que copiaste de Google Drive
FILE_ID = '1EQ-lZG9aSzIQhzLE7sfT6_vce1Q1X4Jg'
URL = f'https://google.com{FILE_ID}'
OUTPUT = 'modelo_comprimido.pkl'

def download():
    if not os.path.exists(OUTPUT):
        print("Descargando modelo desde Google Drive...")
        gdown.download(URL, OUTPUT, quiet=False)
        print("Descarga completada.")
    else:
        print("El modelo ya existe localmente.")

if __name__ == '__main__':
    download()