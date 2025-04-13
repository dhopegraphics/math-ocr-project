import os
import tarfile
from tqdm import tqdm
import cv2
import numpy as np
import logging
from .preprocessing import is_blank_image

class Im2LatexDataset:
    def __init__(self, data_dir, max_formula_len=150):
        self.data_dir = data_dir
        self.max_formula_len = max_formula_len
        self.logger = logging.getLogger(__name__)
        self.formulas = self._load_formulas()
        self.train, self.val, self.test = self._load_splits()
        
    def _load_formulas(self):
        """Load formulas with encoding fallback"""
        formula_file = os.path.join(self.data_dir, 'im2latex_formulas.lst')
        self.logger.info(f"Loading formulas from {formula_file}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(formula_file, 'r', encoding=encoding) as f:
                    formulas = [formula.strip() for formula in f.readlines()]
                self.logger.info(f"Successfully loaded using {encoding} encoding")
                return formulas
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode {formula_file} with any supported encoding")
    
    def _load_splits(self):
        """Load dataset splits with error handling"""
        # Extract images if needed
        if not os.path.exists(os.path.join(self.data_dir, 'formula_images')):
            self.logger.info("Extracting images...")
            try:
                with tarfile.open(os.path.join(self.data_dir, 'formula_images.tar.gz'), 'r:gz') as tar:
                    tar.extractall(path=self.data_dir)
            except Exception as e:
                self.logger.error(f"Failed to extract images: {str(e)}")
                raise
        
        splits = {}
        for split in ['train', 'validate', 'test']:
            split_file = os.path.join(self.data_dir, f'im2latex_{split}.lst')
            self.logger.info(f"Processing {split} split from {split_file}")
            
            try:
                with open(split_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                with open(split_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            
            pairs = []
            for line in tqdm(lines, desc=f"Loading {split} images"):
                try:
                    parts = line.strip().split()
                    formula_idx = int(parts[0])
                    img_name = parts[1]
                    formula = self.formulas[formula_idx]
                    img_path = os.path.join(self.data_dir, 'formula_images', f'{img_name}.png')
                    
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None and not is_blank_image(img):
                            pairs.append((img_path, formula))
                        else:
                            self.logger.debug(f"Skipping blank/corrupted image: {img_path}")
                except Exception as e:
                    self.logger.warning(f"Error processing line: {line.strip()}. Error: {str(e)}")
                    continue
            
            splits[split] = pairs
            self.logger.info(f"Loaded {len(pairs)} valid {split} samples")
        
        return splits['train'], splits['validate'], splits['test']