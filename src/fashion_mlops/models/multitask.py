import copy
import time

import mlflow
import numpy as np
import torch
from torch import nn
from torchvision import models

from fashion_mlops.models.train import compute_metrics


class MultiTaskResNet(nn.Module):
    """
    ResNet18 preentrenada en ImageNet compartida por varias tareas: el backbone extrae un único
    vector de 512 características por imagen y cada tarea tiene su propia capa de salida.
    `num_classes` es un diccionario tarea -> número de clases. Devuelve un diccionario
    tarea -> logits.
    """

    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

        self.heads = nn.ModuleDict({task: nn.Linear(in_features, n) for task, n in num_classes.items()})

    def forward(self, x):
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


def train_one_epoch_multitask(model, loader, criterion, optimizer, device, tasks):
    """
    Entrena una época. La pérdida es la suma de las pérdidas de cada tarea (todas con el mismo
    peso). Las etiquetas llegan como un tensor (batch, tareas) en el orden de `tasks`.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = sum(criterion(outputs[task], labels[:, i]) for i, task in enumerate(tasks))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_multitask(model, loader, device, tasks):
    """
    Devuelve dos diccionarios tarea -> array: las etiquetas reales y las predichas.
    """
    model.eval()
    y_true = {task: [] for task in tasks}
    y_pred = {task: [] for task in tasks}

    for images, labels in loader:
        outputs = model(images.to(device))
        for i, task in enumerate(tasks):
            y_true[task].append(labels[:, i].numpy())
            y_pred[task].append(outputs[task].argmax(dim=1).cpu().numpy())

    return ({task: np.concatenate(v) for task, v in y_true.items()},
            {task: np.concatenate(v) for task, v in y_pred.items()})


def fit_multitask(model, train_loader, val_loader, criterion, optimizer, device, epochs, tasks):
    """
    Entrena el modelo `epochs` épocas evaluando cada tarea en validación al final de cada una.
    Si hay un run de MLflow activo, registra las métricas por época. El criterio para quedarse
    con la mejor época es la media del F1-macro de todas las tareas (`val_f1_macro_mean`).
    Devuelve el historial y deja en el modelo los pesos de la mejor época.
    """
    history = []
    best_score, best_weights = -1, None

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_one_epoch_multitask(model, train_loader, criterion, optimizer, device, tasks)
        y_true, y_pred = predict_multitask(model, val_loader, device, tasks)

        metrics = {"train_loss": train_loss}
        for task in tasks:
            metrics.update({f"val_{task}_{k}": v for k, v in compute_metrics(y_true[task], y_pred[task]).items()})
        metrics["val_f1_macro_mean"] = float(np.mean([metrics[f"val_{task}_f1_macro"] for task in tasks]))

        history.append({"epoch": epoch, **metrics})
        f1_by_task = " | ".join(f"{task} {metrics[f'val_{task}_f1_macro']:.4f}" for task in tasks)
        print(f"Época {epoch}: loss {train_loss:.4f} | val F1-macro: {f1_by_task} | media {metrics['val_f1_macro_mean']:.4f} | {time.time() - start:.0f}s")

        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=epoch)

        # Guardar los pesos de la mejor época
        if metrics["val_f1_macro_mean"] > best_score:
            best_score = metrics["val_f1_macro_mean"]
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return history
