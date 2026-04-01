"""
One Piece Character Detection avec YOLOv8

Fonctionnalités :
- Entraînement
- Évaluation
- Prédiction (image, dossier, vidéo)
- Export du modèle

Auteur: ton équipe
"""

from ultralytics import YOLO
import argparse
import os


# =========================
# CONFIGURATION
# =========================
DEFAULT_MODEL = "yolov8s.pt"
DATA_CONFIG = "data.yaml"
PROJECT_NAME = "onepiece_detector"


# =========================
# TRAIN
# =========================
def train(args):
    print("🚀 Lancement de l'entraînement...")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device
    )

    print("✅ Entraînement terminé !")


# =========================
# VALIDATION
# =========================
def validate(args):
    print("📊 Évaluation du modèle...")

    model = YOLO(args.weights)

    metrics = model.val()
    print(metrics)

    print("✅ Évaluation terminée !")


# =========================
# PREDICTION
# =========================
def predict(args):
    print("🔍 Lancement des prédictions...")

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        show=args.show
    )

    print("📁 Résultats sauvegardés dans runs/detect/predict/")
    print("✅ Prédiction terminée !")


# =========================
# EXPORT
# =========================
def export_model(args):
    print("💾 Export du modèle...")

    model = YOLO(args.weights)

    model.export(format=args.format)

    print(f"✅ Modèle exporté au format {args.format}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 - One Piece Detector")

    subparsers = parser.add_subparsers(dest="mode", help="Mode d'exécution")

    # ---- TRAIN ----
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    train_parser.add_argument("--data", type=str, default=DATA_CONFIG)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--imgsz", type=int, default=512)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--device", type=str, default="cpu")  # "0" pour GPU
    train_parser.add_argument("--name", type=str, default=PROJECT_NAME)

    # ---- VALIDATE ----
    val_parser = subparsers.add_parser("val")
    val_parser.add_argument("--weights", type=str, required=True)

    # ---- PREDICT ----
    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--weights", type=str, required=True)
    pred_parser.add_argument("--source", type=str, required=True)
    pred_parser.add_argument("--conf", type=float, default=0.25)
    pred_parser.add_argument("--show", action="store_true")

    # ---- EXPORT ----
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--weights", type=str, required=True)
    export_parser.add_argument("--format", type=str, default="onnx")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "val":
        validate(args)

    elif args.mode == "predict":
        predict(args)

    elif args.mode == "export":
        export_model(args)

    else:
        print("❌ Mode invalide. Utilise: train | val | predict | export")


if __name__ == "__main__":
    main()