# Test that the model builds correctly
from src.models.model import MathOCRModel
model = MathOCRModel(vocab_size=100, max_formula_len=50)
train_model, enc_model, dec_model = model.build_model()
print("Training model summary:")
train_model.summary()