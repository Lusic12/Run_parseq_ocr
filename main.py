import torch
from PIL import Image
import sys
import statistics
import os
from tqdm import tqdm
sys.path.append('/content/Run_parseq_ocr/parseq')

from parseq.strhub.data.module import SceneTextDataModule
from parseq.strhub.models.utils import load_from_checkpoint

class PARSeqPredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.parseq, self.img_transform = self.load_model_parseq(checkpoint_path, device)

    def load_model_parseq(self, checkpoint_path, device):
        parseq = load_from_checkpoint(checkpoint_path).eval().to(device)
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
        return parseq, img_transform

    @torch.inference_mode()
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pred_text, confidence = self.predict_parseq(image)
        return pred_text, confidence

    @torch.inference_mode()
    def predict_parseq(self, image):
        image = self.img_transform(image).unsqueeze(0).to(self.device)
        p = self.parseq(image).softmax(-1)
        pred, p = self.parseq.tokenizer.decode(p)
        return (pred, statistics.mean(p[0].tolist()))

def predict_multiple(checkpoint_path, image_dir, device='cuda'):
    predictor = PARSeqPredictor(checkpoint_path, device)
    results = []

    # Lấy danh sách tất cả các file ảnh trong thư mục
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        try:
            pred_text, confidence = predictor.predict(image_path)
            results.append({
                'file': image_file,
                'predicted_text': pred_text,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    # In kết quả
    for result in results:
        print(f"File: {result['file']}")
        print(f"Văn bản dự đoán: {result['predicted_text']}")
        print(f"Độ tin cậy: {result['confidence']:.4f}")
        print("-" * 50)

    return results

