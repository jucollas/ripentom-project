
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import time
import joblib

def training_ml_models(models, X_train, y_train, X_val, y_val):
  scaler = StandardScaler()
  X_train_sc = scaler.fit_transform(X_train)
  X_val_sc   = scaler.transform(X_val)
  resultados_ml = {}

  print(f"{'Modelo':<22} {'Accuracy':>10} {'F1-macro':>10} {'Tiempo':>10}")
  print("─" * 56)

  for nombre, modelo in models.items():
      # SVM y KNN usan datos escalados
      usa_escala = nombre in ["SVM (RBF)", "KNN (k=5)"]
      Xtr = X_train_sc if usa_escala else X_train
      Xv  = X_val_sc   if usa_escala else X_val

      t0 = time.time()
      modelo.fit(Xtr, y_train)
      preds = modelo.predict(Xv)
      elapsed = time.time() - t0

      acc = accuracy_score(y_val, preds)
      f1  = f1_score(y_val, preds, average="macro")
      resultados_ml[nombre] = {"acc": acc, "f1": f1, "preds": preds}

      print(f"{nombre:<22} {acc:>10.3f} {f1:>10.3f} {elapsed:>9.1f}s")
      
      joblib.dump(models["Random Forest"], "modelo_tomate_rf.joblib")
      joblib.dump(scaler, "scaler.joblib")
      
      return resultados_ml
  
def report(results):
  CLASS_NAME = ["damaged", "unripe", "ripe"]
  mejor_ml = max(results, key=lambda k: results[k]["acc"])
  print(f"\nMejor modelo ML: {mejor_ml}")
  print(classification_report(y_val, results[mejor_ml]["preds"],
                              target_names=CLASS_NAME))

  print("Modelos guardados ✓")
  
if __name__ == "__main__":
  from src.agent.model_ml import (models, )
  from src.evironment.feature_extraction import split_feature_extraction
  from pathlib import Path
  split_path = Path("data/processed/") 
  X_train, y_train, X_val, y_val = split_feature_extraction(split_path)
  results = training_ml_models(models, X_train, y_train, X_val, y_val)
  report(results)
  


