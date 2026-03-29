# %% [markdown]
# ## Imports

# %%
import re
from collections import Counter

import os

sys.path.append(os.path.abspath("../numpy"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import Data
from layers import DenseLayer
from activation import ReLUActivation, SoftmaxActivation
from neuralnet import NeuralNetwork
from losses import CategoricalCrossEntropy
from metrics import accuracy

# %% [markdown]
# ## Ler os datasets

# %%
train_df = pd.read_csv("../dataset_distribuido_teste.csv", sep=";")
test_df = pd.read_csv("../subm2.csv", sep=";")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

display(train_df.head())
display(test_df.head())

# %% [markdown]
# ## Limpeza de texto

# %%
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def add_bigrams(text):
    words = text.split()
    bigrams = [words[i] + "_" + words[i+1] for i in range(len(words)-1)]
    return " ".join(words + bigrams)


cleanText=False ####################################### mudar 

if cleanText:
    train_df["clean_text"] = train_df["Text"].apply(clean_text)
    test_df["clean_text"] = test_df["Text"].apply(clean_text)
    print("com texto limpo")
else:
    train_df["clean_text"] = train_df["Text"]
    test_df["clean_text"] = test_df["Text"]
    print("sem texto limpo")

display(train_df[["Text", "clean_text", "Label"]].head())

# %% [markdown]
# ## Ver classes

# %%
print("Classes:")
print(train_df["Label"].value_counts())

# %% [markdown]
# ## Construir vocabulário
# 
# Bag of Words

# %%
def build_vocab(texts, max_features=20000, min_freq=2):
    counter = Counter()
    
    for text in texts:
        counter.update(text.split())
    
    vocab_items = [
        word for word, freq in counter.items()
        if freq >= min_freq
    ]
    
    vocab_items = sorted(
        vocab_items,
        key=lambda word: counter[word],
        reverse=True
    )[:max_features]
    
    vocab = {word: idx for idx, word in enumerate(vocab_items)}
    return vocab



# %% [markdown]
# ## Transformar texto em Bag of Words

# %%
def text_to_bow(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    
    for i, text in enumerate(texts):
        for word in text.split():
            if word in vocab:
                X[i, vocab[word]] += 1.0
                
    return X


# %% [markdown]
# ## Transformar texto em TFIDF

# %%
def compute_idf(texts, vocab):
    n_docs = len(texts)
    df = np.zeros(len(vocab), dtype=np.float32)

    for text in texts:
        seen_words = set()
        for word in text.split():
            if word in vocab and word not in seen_words:
                df[vocab[word]] += 1
                seen_words.add(word)

    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return idf

def text_to_tfidf(texts, vocab, idf):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)

    for i, text in enumerate(texts):
        words = text.split()
        if len(words) == 0:
            continue

        for word in words:
            if word in vocab:
                X[i, vocab[word]] += 1.0

        X[i] = X[i] / len(words)
        X[i] = X[i] * idf

        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] = X[i] / norm


    return X


# %% [markdown]
# ## Codificar labels

# %%
def fit_label_encoder(labels):
    classes = sorted(labels.unique())
    label_to_idx = {label: i for i, label in enumerate(classes)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

def encode_labels(labels, label_to_idx):
    return np.array([label_to_idx[label] for label in labels], dtype=np.int32)

def one_hot_encode(y, num_classes):
    y_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
    y_onehot[np.arange(len(y)), y] = 1.0
    return y_onehot


# %% [markdown]
# ## Split treino / validação

# %%
def train_val_split(X, y, val_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - val_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

texts = train_df["clean_text"].to_numpy()
labels = train_df["Label"].to_numpy()

X_train_text, X_val_text, y_train_text, y_val_text = train_val_split(texts, labels, val_size=0.2, seed=42)

print("X_train:", X_train_text.shape)
print("X_val:", X_val_text.shape)
print("y_train:", y_train_text.shape)
print("y_val:", y_val_text.shape)

# %%
def fit_vectorizer(texts_train, use_tfidf=True, use_bigrams=False, max_features=15000, min_freq=2):
    if use_bigrams:
        texts_train = np.array([add_bigrams(t) for t in texts_train])

    vocab = build_vocab(texts_train, max_features=max_features, min_freq=min_freq)

    vectorizer = {
        "use_tfidf": use_tfidf,
        "use_bigrams": use_bigrams,
        "vocab": vocab,
        "idf": None
    }

    if use_tfidf:
        vectorizer["idf"] = compute_idf(texts_train, vocab)

    return vectorizer

# %%
def transform_texts(texts, vectorizer):
    if vectorizer["use_bigrams"]:
        texts = np.array([add_bigrams(t) for t in texts])

    vocab = vectorizer["vocab"]

    if vectorizer["use_tfidf"]:
        idf = vectorizer["idf"]
        return text_to_tfidf(texts, vocab, idf)
    else:
        return text_to_bow(texts, vocab)

# %%
def run_experiment(
    model_name,
    texts_train,
    texts_val,
    y_train,
    y_val,
    use_tfidf=True,
    use_bigrams=False,
    max_features=15000,
    min_freq=2,
    epochs=50,
    batch_size=32,
    learning_rate=None,
    momentum=0.9
):
    vectorizer = fit_vectorizer(
        texts_train,
        use_tfidf=use_tfidf,
        use_bigrams=use_bigrams,
        max_features=max_features,
        min_freq=min_freq
    )

    X_train = transform_texts(texts_train, vectorizer)
    X_val = transform_texts(texts_val, vectorizer)

    train_data = Data(X_train, y_train)
    val_data = Data(X_val, y_val)

    if model_name == "logreg":
        if learning_rate is None:
            learning_rate = 0.05

        model = NeuralNetwork(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            verbose=False,
            loss=CategoricalCrossEntropy,
            metric=accuracy
        )
        model.add(DenseLayer(y_train.shape[1], input_shape=(X_train.shape[1],), l2_lambda=1e-4))
        model.add(SoftmaxActivation())

    elif model_name == "dnn":
        if learning_rate is None:
            learning_rate = 0.1

        model = NeuralNetwork(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            verbose=False,
            loss=CategoricalCrossEntropy,
            metric=accuracy
        )
        model.add(DenseLayer(128, input_shape=(X_train.shape[1],), l2_lambda=1e-4))
        model.add(ReLUActivation())
        model.add(DenseLayer(64, l2_lambda=1e-4))
        model.add(ReLUActivation())
        model.add(DenseLayer(y_train.shape[1], l2_lambda=1e-4))
        model.add(SoftmaxActivation())

    else:
        raise ValueError("model_name must be 'logreg' or 'dnn'")

    model.fit(train_data, val_dataset=val_data, early_stopping=True, patience=5)

    preds = model.predict(val_data)
    acc = accuracy(y_val, preds)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "val_accuracy": acc,
        "history": model.history
    }

# %%
texts = train_df["clean_text"].to_numpy()
labels = train_df["Label"].to_numpy()

X_train_text, X_val_text, y_train_text, y_val_text = train_val_split(
    texts, labels, val_size=0.2, seed=42
)

label_to_idx, idx_to_label = fit_label_encoder(train_df["Label"])

y_train_idx = encode_labels(y_train_text, label_to_idx)
y_val_idx = encode_labels(y_val_text, label_to_idx)

num_classes = len(label_to_idx)
y_train = one_hot_encode(y_train_idx, num_classes)
y_val = one_hot_encode(y_val_idx, num_classes)

# %% [markdown]
# ## Modelos
# 

# %%
results = {}

results["logreg_bow"] = run_experiment(
    model_name="logreg",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=False,
    use_bigrams=False
)

results["logreg_tfidf"] = run_experiment(
    model_name="logreg",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=True,
    use_bigrams=False
)

results["logreg_tfidf_bigram"] = run_experiment(
    model_name="logreg",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=True,
    use_bigrams=True
)


results["dnn_bow"] = run_experiment(
    model_name="dnn",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=False,
    use_bigrams=False
)

results["dnn_tfidf"] = run_experiment(
    model_name="dnn",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=True,
    use_bigrams=False
)



results["dnn_tfidf_bigram"] = run_experiment(
    model_name="dnn",
    texts_train=X_train_text,
    texts_val=X_val_text,
    y_train=y_train,
    y_val=y_val,
    use_tfidf=True,
    use_bigrams=True
)

# %% [markdown]
# ## Graficos

# %%
def plot_history(history, title="Model"):
    epochs = list(history.keys())

    train_loss = [history[e]["loss"] for e in epochs]
    val_loss = [history[e]["val_loss"] for e in epochs]
    train_acc = [history[e]["metric"] for e in epochs]
    val_acc = [history[e]["val_metric"] for e in epochs]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
plot_history(results["logreg_bow"]["history"], title="LogReg + BoW")
plot_history(results["logreg_tfidf"]["history"], title="LogReg + TF-IDF")
plot_history(results["logreg_tfidf_bigram"]["history"], title="LogReg + TF-IDF + Bigrams")
plot_history(results["dnn_bow"]["history"], title="DNN + BoW")
plot_history(results["dnn_tfidf"]["history"], title="DNN + TF-IDF")
plot_history(results["dnn_tfidf_bigram"]["history"], title="DNN + TF-IDF + Bigrams")

# %% [markdown]
# ## Escolher o melhor modelo

# %%
for name, result in results.items():
    print(name, "->", result["val_accuracy"])

best_model_name = max(results, key=lambda k: results[k]["val_accuracy"])
best_result = results[best_model_name]

best_model = best_result["model"]
best_vectorizer = best_result["vectorizer"]

print("Best model:", best_model_name)
print("Validation accuracy:", best_result["val_accuracy"])

# %% [markdown]
# # TESTE COM SUMB1 COM LABELS

# %%
val_test_df = pd.read_csv("../subm1_labels_revealed.csv", sep=";")

display(val_test_df.head())

# %%
val_test_df["clean_text"] = val_test_df["Text"]
#val_test_df["clean_text"] = val_test_df["Text"].apply(clean_text)
texts_val_test = val_test_df["clean_text"].to_numpy()


# %%
test_results = {}

true_idx = np.array([label_to_idx[label] for label in val_test_df["Label"]])

for name, result in results.items():
    model = result["model"]
    vectorizer = result["vectorizer"]

    X_val_test = transform_texts(texts_val_test, vectorizer)

    val_test_data = Data(X_val_test, y=None)

    preds = model.predict(val_test_data)
    pred_idx = np.argmax(preds, axis=1)

    acc = np.mean(pred_idx == true_idx)

    test_results[name] = acc
    print(f"{name}: {acc:.4f}")

# %%
pred_labels = [idx_to_label[i] for i in pred_idx]
for i in range(10):
    print("Texto:", val_test_df["Text"].iloc[i][:80])
    print("Real:", val_test_df["Label"].iloc[i])
    print("Pred:", pred_labels[i])
    print("---")

# %% [markdown]
# # Predict Submision

# %%
model_name = "logreg_bow"   # escolhe aqui
#model_name = max(results, key=lambda k: results[k]["val_accuracy"])

# %%
selected_result = results[model_name]

model = selected_result["model"]
vectorizer = selected_result["vectorizer"]

print("Using model:", model_name)

# %%
X_test = transform_texts(test_df["clean_text"].to_numpy(), vectorizer)

# %%
test_data = Data(X_test, y=None)

# %%
pred_probs = model.predict(test_data)
pred_idx = np.argmax(pred_probs, axis=1)

# %%
pred_labels = [idx_to_label[i] for i in pred_idx]

# %%
submission = pd.DataFrame({
    "ID": test_df["ID"],
    "Label": pred_labels
})

submission.to_csv("subm2-g6-MEI-A.csv", sep=";", index=False)
display(submission.head())

# %%



