import argparse
import cv2
from src.models.train import train_model
from src.models.predict import Predictor
from src.data.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description='Math OCR System')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data_dir', required=True)
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=50)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--image_path', required=True)
    predict_parser.add_argument('--model_path', default='saved_models/best_model.keras')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir, args.batch_size, args.epochs)
    elif args.command == 'predict':
        # Load tokenizer (you'll need to save this during training)
        tokenizer = Tokenizer()
        tokenizer.load('saved_models/tokenizer.json')
        
        predictor = Predictor(args.model_path, tokenizer)
        image = cv2.imread(args.image_path)
        prediction = predictor.predict(image)
        print(f"Predicted LaTeX: {prediction}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()