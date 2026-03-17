# Projeto de Aprendizagem Profunda 2025/2026
## Grupo 6  
##### Vicente Martins, Diogo Rodrigues, Rui Gonçalves, Simão Alves, Gabriel Antunes - pg{58812, 60244, 59847, 58810, 60259}@alunos.uminho.pt 


Este repositório contém o código, os datasets e os modelos desenvolvidos para a Tarefa de Classificação de Textos. O objetivo principal deste projeto é distinguir parágrafos científicos (com extensão rigorosa de 80 a 120 palavras) escritos por Humanos daqueles gerados por 4 famílias de modelos de IA: OpenAI, Anthropic, Google e Meta.

O repositório está estruturado para responder às duas submissões exigidas:
* **Submissão A:** Implementação de raiz (from scratch) usando `numpy`.
* **Submissão B:** Implementação recorrendo à framework `PyTorch`.

---

## 📂 Estrutura do Repositório

### 1. Manipulação e Limpeza de Dados (Raiz do Repositório)
Esta secção contém todos os ficheiros relativos à **Tarefa 1** (Recolha e Pré-processamento de Dados).

* 📄 **`dataset_manipulation.ipynb`**: O notebook central de *Data Wrangling*. É aqui que os dados em bruto são filtrados (garantindo a regra das 80-120 palavras), limpos de caracteres inválidos, misturados aleatoriamente e formatados com os IDs finais (`D1-XXXX`).
* 📄 **`testdata.py`**: Script de segurança para validar a estrutura, tamanho e formatação dos ficheiros CSV antes do treino.
* 📊 **`dataset_final.csv`**: O dataset final, imaculado e balanceado, usado para treinar os modelos.
* 📊 **`dataset_ai.csv`, `dataset_human.csv`, `dataset_human_limpo.csv`, `dataset-exemplos.csv`**: Ficheiros de dados parciais e em bruto que serviram de base para a construção do dataset final.
* 📊 **`subm1.csv`**: Ficheiro fornecido pelo docente com os casos de teste para classificação.
* 📊 **`subm1-g6-MEI-B.csv`**: Ficheiro gerado pelo nosso modelo PyTorch contendo as previsões finais para avaliação.

---

### 2. Pasta `numpy/` (Submissão A - Implementação Própria)
Contém a implementação de algoritmos de Machine Learning (Regressão Logística e Redes Neuronais) construídos inteiramente do zero, sem recurso a frameworks de Deep Learning.

* 📁 **`logistic_regression/`**: Pasta dedicada ao modelo baseline de regressão logística.
* 📄 **`layers.py`**: Definição matemática das camadas da rede neuronal (ex: Linear).
* 📄 **`activation.py`**: Funções de ativação (ReLU, Sigmoid, Softmax, etc.) e respetivas derivadas.
* 📄 **`losses.py`**: Funções de perda.
* 📄 **`optimizer.py`**: Algoritmos de otimização para a atualização dos pesos.
* 📄 **`metrics.py`**: Funções para cálculo de métricas de avaliação (Accuracy, etc.).
* 📄 **`neuralnet.py`**: Classe principal que orquestra a montagem das camadas e o ciclo de *forward/backward pass*.
* 📄 **`data.py`**: Processamento e divisão dos dados em *batches* de treino e validação.
* 📓 **`notebook.ipynb` & `notebook2.ipynb`**: Notebooks demonstrativos que importam estes módulos, treinam os modelos criados do zero e avaliam os seus resultados no dataset.

---

### 3. Pasta `pytorch/` (Submissão B - PyTorch)
Contém a implementação de arquiteturas de Deep Learning otimizadas, focadas no processamento de sequências de texto.

* 📓 **`PyTorch_Models.ipynb`**: Notebook completo contendo todo o *pipeline* em PyTorch.
  * Criação do vocabulário e codificação/padding do texto.
  * Implementação e treino de três arquiteturas distintas: **RNN**, **LSTM** e **GRU**.
  * Avaliação detalhada de *Overfitting* (com gráficos de Loss/Accuracy) e matrizes de confusão.
  * Geração do ficheiro final de submissão com o modelo mais performante (LSTM).