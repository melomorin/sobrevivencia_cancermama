#!/usr/bin/env bash
# Salir inmediatamente si un comando falla
set -o errexit

# Actualizar el gestor de paquetes e instalar compiladores necesarios
apt-get update && apt-get install -y gfortran libblas-dev liblapack-dev

# Instalar las dependencias de Python
pip install --upgrade pip
pip install -r requirements.txt
