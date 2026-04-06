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

### 2. 📁 Pasta `Subm2/` (Submissões Oficiais)
Pasta central contendo os ficheiros exigidos para a avaliação, nomeados de acordo com a regra oficial `subm2-g<ng>-<curso>-<sub>`.

* 📄 **`subm2-g6-MEI-A.ipynb`**: Notebook com o código executável da **Submissão A** (implementação de raiz utilizando *numpy*).
* 📊 **`subm2-g6-MEI-A.csv`**: Ficheiro com as previsões finais geradas pela implementação própria.
* 📄 **`subm2-g6-MEI-B.ipynb`**: Notebook com o código executável da **Submissão B** (implementação utilizando a framework *PyTorch*).
* 📊 **`subm2-g6-MEI-B.csv`**: Ficheiro com as previsões finais geradas pelos modelos desenvolvidos em PyTorch.

### 3. 📁 Pasta `Subm3/` (Submissões Oficiais)
Pasta central contendo os ficheiros exigidos para a avaliação, nomeados de acordo com a regra oficial `subm3-g<ng>-<curso>-<sub>`.

* 📄 **`subm3-g6-MEI-A.ipynb`**: Notebook com o código executável da **Submissão A** (implementação de raiz utilizando *numpy*).
* 📊 **`subm3-g6-MEI-A.csv`**: Ficheiro com as previsões finais geradas pela implementação própria.
* 📄 **`subm3-g6-MEI-B.ipynb`**: Notebook com o código executável da **Submissão B** (implementação utilizando a framework *PyTorch*).
* 📊 **`subm3-g6-MEI-B.csv`**: Ficheiro com as previsões finais geradas pelos modelos desenvolvidos em PyTorch.

### 4. 📁 Pasta `numpy/` (Motor de Implementação Própria)
Contém a base de código modular construída do zero para suportar a Submissão A.
* 📄 **Scripts Core (`layers.py`, `activation.py`, `losses.py`, `optimizer.py`, `metrics.py`, `neuralnet.py`, `data.py`)**: Implementação matemática das redes neuronais, cálculo de gradientes, funções de custo e processamento de batches.

### 5. Ficheiros na Raiz (Dados e Pré-processamento)
Contém o material relativo à Tarefa 1 (Recolha e Tratamento de Dados).
* 📄 **`dataset_manipulation.ipynb`**: Notebook central de *Data Wrangling* responsável pela filtragem, limpeza, baralhamento (*shuffle*) e formatação dos IDs.
* 📄 **`testdata.py`**: Script de segurança para validar a estrutura e tamanho dos ficheiros CSV.
* 📊 **`dataset_final.csv`**: O dataset final, imaculado e balanceado, usado para treinar os modelos finais da subm1.
* 📊 **`subm1.csv`**: Ficheiro fornecido pelo docente com os casos de teste para classificação (sem labels).
* 📊 **`dataset_final_v2.csv`**: O nosso dataset_final mas com o subm1 com labels reveladas.
* 📊 **`subm1_predicetd_test.csv`**: Ficheiro previsto com o modelo para comparar com o subm1_labels revealed.csv para podermos testar (foi treinado com o dataset_final ou dataset_distribuido_teste)
* 📊 **`subm1_labels_revealed.csv`**: Ficheiro fornecido pelo docente com as labels reveladas do subm1.csv
* 📊 **`dataset_distribuido.csv`**: dataset_final_v2.csv mas com os textos humanos a ser 1/3 dos casos totais.
* 📊 **`dataset_distribuido_igualmente.csv`**: dataset_final_v2.csv mas com os textos humanos e das AIs com a mesma quantidade de ocorrências.
* 📊 **`dataset_distribuido_teste.csv`**: dataset_final_v1.csv mas com os textos humanos a ser 1/3 dos casos totais.

* 📊 **`dataset_wiki_humanos.csv`**: conjunto de dados com conteúdos humanos provenientes da Wiki.
* 📊 **`dataset_ai_anthropic.csv`**: conjunto de dados gerado pela Anthropic, contendo as primeiras 10 palavras de cada entrada do dataset_wiki_humanos.csv.
* 📊 **`dataset_ai_google.csv`**: conjunto de dados gerado pela Anthropic, contendo as primeiras 10 palavras de cada entrada do dataset_wiki_humanos.csv.
* 📊 **`dataset_ai_openai.csv`**: conjunto de dados gerado pela Anthropic, contendo as primeiras 10 palavras de cada entrada do dataset_wiki_humanos.csv.
* 📊 **`dataset_master_final.csv`**: conjunto de dados que combina e mistura os ficheiros dataset_wiki_humanos.csv, dataset_ai_anthropic.csv, dataset_ai_google.csv e dataset_ai_openai.csv.
* 📊 **`dataset_train.csv`**: conjunto de treino obtido a partir do dataset_master_final.csv, excluindo 500 dados de cada classe.
* 📊 **`dataset_test_balanced.csv`**: conjunto de teste equilibrado, contendo 500 dados de cada classe do dataset_master_final.csv.
* 📊 **Outros CSVs (`dataset_ai.csv`, `dataset_human.csv`, etc.)**: Ficheiros de dados parciais em bruto que serviram de base.
