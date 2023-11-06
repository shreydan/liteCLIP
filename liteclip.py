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
        self.model.load_state_dict(torch.load(self.config.state_dict_path))
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
    def _prepare_image(self,image):
        try:
            img = Image.open(image).convert('RGB')
        except Exception:
            raise Exception('provide a valid path for the image')
        img = self._img_tfms(img)
        img = torch.unsqueeze(img, 0)

        return img
    
    
    @torch.no_grad()
    def _prepare_text(self,labels):

        text_inputs = self.tokenizer.batch_encode_plus(
            labels,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        return text_inputs

    
    @torch.no_grad()
    def predict(self,image:str, labels: list[str],top_k:int=5):

        assert len(labels) >= 2, "provide atleast 2 labels"

        if len(labels) < top_k:
            top_k = len(labels)

        img = self._prepare_image(image)
        text = self._prepare_text(labels)

        logits,_ = self.model((img,text))
        
        print('logits',logits)
        logits = torch.flatten(logits)
        probabilities = torch.softmax(logits,dim=0)
        values,indices = torch.topk(probabilities,k=top_k)
        values = [v.item() for v in values]
        indices = [i.item() for i in indices]
        
        result = [(labels[i],v) for v,i in zip(values,indices)]

        return result