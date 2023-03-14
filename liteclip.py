import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from transformers import AutoTokenizer
from model import CLIP
from config import Config    


class ZeroShotPipeline:

    def __init__(self,):

        self.config = Config
        self.model = CLIP(self.config)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder)
        
        self._img_tfms = T.Compose([
            T.Resize(224,interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])

    @torch.no_grad()
    def _get_image_embedding(self,image):
        try:
            img = Image.open(image).convert('RGB')
        except Exception:
            raise Exception('provide a valid path for the image')
        img = self._img_tfms(img)
        img = torch.unsqueeze(img, 0)

        embeddings = self.model.image_encoder(img)
        embeddings = self.model.img_projection(embeddings)
        return embeddings
    
    @torch.no_grad()
    def _get_text_embeddings(self,labels):

        assert len(labels) >= 2, "provide atleast 2 labels"

        text_inputs = self.tokenizer.batch_encode_plus(
            labels,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        embeddings = self.model.text_encoder(text_inputs)
        embeddings = self.model.txt_projection(embeddings)

        return embeddings

    
    @torch.no_grad()
    def predict(self,image:str, labels: list[str]):

        image_embeddings = self._get_image_embedding(image)
        
        text_embeddings = self._get_text_embeddings(labels)
        
        logits = text_embeddings @ image_embeddings.T
        
        logits = torch.flatten(logits)
        probabilities = torch.softmax(logits,dim=0)
        
        return list(zip(labels,probabilities))