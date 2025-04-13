from src.data.preprocessing import load_im2latex_dataset
from src.models.predict import MathOCR

# Load first N samples from dataset
images, labels = load_im2latex_dataset('data/raw/IM2LATEX', split='train')


# Use first sample
image, label = images[0], labels[0]

# Run OCR model
ocr = MathOCR()
pred = ocr.predict(image)

print(f"📌 Actual LaTeX:    {label}")
print(f"🤖 Predicted LaTeX: {pred}")