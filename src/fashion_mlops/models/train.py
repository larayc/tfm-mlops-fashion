import copy
import time

import mlflow
import numpy as np
import torch
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score



def get_device():
    """
    Devuelve el dispositivo de cómputo disponible: GPU de Apple (MPS), GPU NVIDIA (CUDA) o CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes, freeze_backbone=True):
    """
    Crea una ResNet18 preentrenada en ImageNet y sustituye la última capa por una nueva con
    `num_classes` salidas. Con `freeze_backbone` solo se entrena esa última capa.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = not freeze_backbone

    # La capa nueva siempre se entrena
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def unfreeze_model(model):
    """
    Descongela todas las capas del modelo para hacer fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


def compute_metrics(y_true, y_pred):
    """
    Calcula las métricas de clasificación usadas en el proyecto.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Entrena el modelo durante una época y devuelve la pérdida media.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """
    Devuelve las etiquetas reales, las predichas y las probabilidades de cada clase.
    """
    model.eval()
    y_true, y_prob = [], []

    for images, labels in loader:
        logits = model(images.to(device))
        y_prob.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    return y_true, y_prob.argmax(axis=1), y_prob


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=None):
    """
    Entrena el modelo `epochs` épocas evaluando en validación al final de cada una.
    Si hay un run de MLflow activo, registra las métricas por época. Devuelve el historial
    y deja en el modelo los pesos de la época con mejor F1-macro en validación.

    Con `patience` se aplica early stopping: el entrenamiento se detiene si el F1-macro en
    validación no mejora durante `patience` épocas seguidas, y `epochs` pasa a ser el máximo.
    """
    history = []
    best_f1, best_weights = -1, None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, _ = predict(model, val_loader, device)
        metrics = {"train_loss": train_loss, **{f"val_{k}": v for k, v in compute_metrics(y_true, y_pred).items()}}

        history.append({"epoch": epoch + 1, **metrics})
        print(
            f"Época {epoch + 1}: loss {train_loss:.4f} | val f1_macro {metrics['val_f1_macro']:.4f} | "
            f"val balanced_acc {metrics['val_balanced_accuracy']:.4f} | {time.time() - start:.0f}s"
        )

        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=epoch + 1)

        # Guardar los pesos de la mejor época
        if metrics["val_f1_macro"] > best_f1:
            best_f1 = metrics["val_f1_macro"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping: el F1-macro en validación no mejora desde hace {patience} épocas.")
            break

    model.load_state_dict(best_weights)

    return history

