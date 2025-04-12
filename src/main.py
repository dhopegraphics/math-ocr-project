import argparse
from src.models.train import train_model
from src.models.predict import MathOCR  # If you build prediction CLI
import os

def main():
    parser = argparse.ArgumentParser(description='Math OCR System CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Train command ---
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    train_parser.add_argument('--dataset_type', type=str, default='IM2LATEX', help='Dataset type')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')

    # --- Predict command ---
    predict_parser = subparsers.add_parser('predict', help='Run OCR on an image')
    predict_parser.add_argument('--image_path', type=str, required=True, help='Path to the image')

    args = parser.parse_args()

    if args.command == 'train':
        print("[INFO] Starting training...")
        model, history = train_model(args.dataset_path, args.dataset_type, args.epochs)
        print("[INFO] Training complete!")

    elif args.command == 'predict':
        ocr = MathOCR()
        latex = ocr.predict(args.image_path)
        print(f"Predicted LaTeX: {latex}")

if __name__ == '__main__':
    main()