import torch

from parseq.strhub.data.module import SceneTextDataModule
from parseq.strhub.models.utils import load_from_checkpoint


def load_model_parseq(device='cuda'):
    parseq = load_from_checkpoint('./weights/rec/best-parseq.ckpt').eval().to(device)
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

def load_model_parseq_author():
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    return parseq, img_transform

