# ===========================
# Makefile for Biomass Project
# ===========================

PYTHON=python

# ---------------------------
# Setup
# ---------------------------
install:
	pip install -r requirements.txt

env:
	conda env create -f environment.yaml

# ---------------------------
# Training
# ---------------------------
train:
	$(PYTHON) -m src.train.train

# ---------------------------
# Prediction
# ---------------------------
predict:
	$(PYTHON) -m src.train.predict

# ---------------------------
# Format / Lint
# ---------------------------
format:
	black src/

lint:
	flake8 src/

# ---------------------------
# Clean temporary files
# ---------------------------
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

# ---------------------------
# Run everything
# ---------------------------
all: install train predict

