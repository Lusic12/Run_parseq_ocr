import torch
from PIL import Image
import sys
import statistics
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

def predict(checkpoint_path, image_path, device='cuda'):
    predictor = PARSeqPredictor(checkpoint_path, device)
    pred_text, confidence = predictor.predict(image_path)
    print(f"Văn bản dự đoán: {pred_text}")
    print(f"Độ tin cậy: {confidence:.4f}")