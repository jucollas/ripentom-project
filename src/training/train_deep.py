import torch
from src.agent.model_deep import (
    data_loader, 
    model_definition, 
    train_epoch, 
    val_epoch
)


def training_model(model, train_loader, val_loader):

    for param in model.features.parameters():
        param.requires_grad = False

    opt1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    hist = {"train_acc": [], "val_acc": [], "loss": []}

    print("=== FASE 1: solo clasificador ===")
    for ep in range(5):
        loss, tr_acc = train_epoch(device, model, criterion, train_loader, opt1)
        val_acc, _, _ = val_epoch(device, model,val_loader)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)
        hist["loss"].append(loss)
        print(
            f"Ep {ep+1}/5 | loss={loss:.3f} | train={tr_acc:.3f} | val={val_acc:.3f}")

    for i, block in enumerate(model.features):
        for param in block.parameters():
            param.requires_grad = (i >= 10)

    opt2 = torch.optim.Adam([
        {"params": [p for b in model.features[10:]
                    for p in b.parameters()], "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 5e-5},
    ])

    best_val, best_preds, best_labels = 0, [], []

    print("=== FASE 2: fine-tuning ===")
    for ep in range(10):
        loss, tr_acc = train_epoch(device, model, criterion, train_loader, opt2)
        val_acc, preds, labels = val_epoch(device, model, val_loader)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(val_acc)
        hist["loss"].append(loss)
        print(
            f"Ep {ep+1}/10 | loss={loss:.3f} | train={tr_acc:.3f} | val={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            best_preds, best_labels = preds, labels
            torch.save(model.state_dict(), "./ripentom_best.pth")
            print(f"  -> Guardado ✓ (val={val_acc:.3f})")
    return best_val, best_labels, best_preds, hist


def report(best_val, best_labels, best_preds, hist):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    CLASES_NOMBRES = ["danado", "inmaduro", "maduro"]

    # Reporte por clase
    print(f"\nMejor val_acc: {best_val:.3f}\n")
    print(classification_report(best_labels,
          best_preds, target_names=CLASES_NOMBRES))

    # Matriz de confusión
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASES_NOMBRES, yticklabels=CLASES_NOMBRES)
    plt.title("Matriz de confusión — RipenTom")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=150)
    plt.show()

    # Curva de entrenamiento
    epochs_r = range(1, len(hist["train_acc"]) + 1)
    plt.figure(figsize=(8, 4))

    plt.plot(epochs_r, hist["train_acc"], label="Train accuracy")
    plt.plot(epochs_r, hist["val_acc"],   label="Val accuracy")
    plt.axvline(5.5, color="gray", linestyle="--", label="Inicio fase 2")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Curva de entrenamiento — RipenTom")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./training_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    
    IMAGES_PROCESSED = Path("data/processed/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    model, criterion = model_definition(device)
    train_loader, val_loader = data_loader(IMAGES_PROCESSED)
    best_val, best_labels, best_preds, hist = training_model(model, train_loader, val_loader)
    report(best_val, best_labels, best_preds, hist)
