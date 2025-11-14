# ================================
# Dockerfile - Biomass Prediction
# ================================

# Base image (leve)
FROM python:3.10-slim

# Evitar buffer no stdout
ENV PYTHONUNBUFFERED=1

# Instalar dependências básicas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt /app/

# Instalar dependências Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar código fonte
COPY src /app/src
COPY setup.py /app/
COPY data /app/data
COPY models /app/models

# Instalar o pacote localmente
RUN pip install -e .

# Pasta de outputs
RUN mkdir -p /app/outputs

# Comando padrão do container
CMD ["bash"]

