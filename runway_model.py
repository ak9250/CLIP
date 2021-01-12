import torch
import clip
from PIL import Image
import runway

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

@runway.command('translate', inputs={'source_imgs': runway.image(description='input image to be translated')}, outputs={'text': runway.text})
def translate(learn, inputs):
    image = transform(inputs['source_imgs']).unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
    return probs

if __name__ == '__main__':
    runway.run(port=8889)
