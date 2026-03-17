# Projeto de Aprendizagem Profunda 2025/2026
## Grupo 6  
##### Vicente Martins, Diogo Rodrigues, Rui Gonçalves, Simão Alves, Gabriel Antunes - pg{58812, 60244, 59847, 58810, 60259}@alunos.uminho.pt 


Este repositório contém o código, os datasets e os modelos desenvolvidos para a Tarefa de Classificação de Textos. O objetivo principal deste projeto é distinguir parágrafos científicos (com extensão rigorosa de 80 a 120 palavras) escritos por Humanos daqueles gerados por 4 famílias de modelos de IA: OpenAI, Anthropic, Google e Meta.

O repositório está estruturado para responder às duas submissões exigidas:
* **Submissão A:** Implementação de raiz (from scratch) usando `numpy`.
* **Submissão B:** Implementação recorrendo à framework `PyTorch`.

---

## 📂 Estrutura do Repositório

### 1. 📁 Pasta `Subm1/` (Submissões Oficiais)
Pasta central contendo os ficheiros exigidos para a avaliação, nomeados de acordo com a regra oficial `subm1-g<ng>-<curso>-<sub>`.

* 📄 **`subm1-g6-MEI-A.ipynb`**: Notebook com o código executável da **Submissão A** (implementação de raiz utilizando *numpy*).
* 📊 **`subm1-g6-MEI-A.csv`**: Ficheiro com as previsões finais geradas pela implementação própria.
* 📄 **`subm1-g6-MEI-B.ipynb`**: Notebook com o código executável da **Submissão B** (implementação utilizando a framework *PyTorch*).
* 📊 **`subm1-g6-MEI-B.csv`**: Ficheiro com as previsões finais geradas pelos modelos desenvolvidos em PyTorch.

### 2. 📁 Pasta `numpy/` (Motor de Implementação Própria)
Contém a base de código modular construída do zero para suportar a Submissão A.
* 📁 **`logistic_regression/`**: Scripts dedicados ao modelo baseline de regressão logística.
* 📄 **Scripts Core (`layers.py`, `activation.py`, `losses.py`, `optimizer.py`, `metrics.py`, `neuralnet.py`, `data.py`)**: Implementação matemática das redes neuronais, cálculo de gradientes, funções de custo e processamento de batches.

### 3. Ficheiros na Raiz (Dados e Pré-processamento)
Contém o material relativo à Tarefa 1 (Recolha e Tratamento de Dados).
* 📄 **`dataset_manipulation.ipynb`**: Notebook central de *Data Wrangling* responsável pela filtragem, limpeza, baralhamento (*shuffle*) e formatação dos IDs.
* 📄 **`testdata.py`**: Script de segurança para validar a estrutura e tamanho dos ficheiros CSV.
* 📊 **`dataset_final.csv`**: O dataset final, imaculado e balanceado, usado para treinar os modelos finais.
* 📊 **`subm1.csv`**: Ficheiro fornecido pelo docente com os casos de teste para classificação (sem labels).
* 📊 **Outros CSVs (`dataset_ai.csv`, `dataset_human.csv`, etc.)**: Ficheiros de dados parciais em bruto que serviram de base.
