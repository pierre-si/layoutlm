#%%
import ast
from io import BytesIO
from logging import basicConfig

from PIL import Image
import requests
from ray import serve
import torch

from layoutlmft.trainers import FunsdTrainer as Trainer
from torchvision import transforms
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)
from detectron2.structures import ImageList
# %%
@serve.deployment(route_prefix="/receipt")
class LayoutModel:
    def __init__(self, model_name = "../output/v2_local_cpu/", num_labels = 55):
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task="ner",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=self.config,
        )

        
    async def __call__(self, starlette_request):
        dict_payload_bytes = await starlette_request.body()
        # batch = ast.literal_eval(dict_payload_bytes)
        batch = eval(dict_payload_bytes) # unsafe
        # pil_image = Image.open(BytesIO(image_payload_bytes))
        # pil_images = [pil_image]  # Our current batch size is one

        input_ids = torch.tensor(batch["input_ids"])
        bbox = torch.tensor(batch["bbox"])
        image = ImageList.from_tensors([torch.tensor(batch["image"])])
        att_mask = torch.tensor(batch["attention_mask"])
        token_type_ids = torch.tensor(batch["token_type_ids"])
        with torch.no_grad():
            y = self.model(input_ids, bbox, image, att_mask, token_type_ids)

        return {"class_index": y["logits"][0].argmax(axis=1).numpy()}
#%%
serve.start()
LayoutModel.deploy()
# %%
