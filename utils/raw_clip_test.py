from tqdm import tqdm
from train_net import extract_from_batch_data
import torch
from models import clip

prompts=['a picture of a person','a picture of an apple','a picture of a car','a picture of a cat','a picture of a laptop']


def test_raw_clip(cfg, logger, train_loader, raw_clip):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = raw_clip.encode_text(text)

    for idx, batch_data in enumerate(tqdm(train_loader)):
        images, labels = extract_from_batch_data(batch_data,
                                                 device)  # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
        image = images[0] # 取每个batch的第一个
        image_features = raw_clip.encode_image(image)
        logits_per_image, logits_per_text = raw_clip(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


