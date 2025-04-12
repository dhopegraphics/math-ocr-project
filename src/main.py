import argparse
from models.predict import MathOCR
from models.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Math OCR System')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('image_path', help='Path to input image')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Training model...")
        model, history = train_model()
        print("Training complete!")
    elif args.command == 'predict':
        ocr = MathOCR()
        result = ocr.predict(args.image_path)
        print(f"Predicted LaTeX: {result}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()