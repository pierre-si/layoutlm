#%%
import ast
from io import BytesIO
from logging import basicConfig

from PIL import Image
import requests
from ray import serve
import torch
from torchvision import transforms
from datasets import load_dataset, load_metric, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)
from detectron2.structures import ImageList
import easyocr

from layoutlmft.trainers import FunsdTrainer as Trainer
from layoutlmft.data.utils import load_image, normalize_bbox
from layoutlmft.data import DataCollatorForKeyValueExtraction
#%%
text_column_name = "tokens"
label_column_name = "ner_tags"
padding = "max_length"
label_to_id = {i: i for i in range(55)}
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
        self.reader = easyocr.Reader(["en"])
        self.data_collator = DataCollatorForKeyValueExtraction(
            self.tokenizer,
            pad_to_multiple_of=8,
            padding=padding,
            max_length=512,
        )

    def tokenize_and_align_labels(self, examples):

        tokenized_inputs = self.tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs

    # async def __call__(self, starlette_request):
    #     dict_payload_bytes = await starlette_request.body()
    #     # batch = ast.literal_eval(dict_payload_bytes)
    #     batch = eval(dict_payload_bytes) # unsafe

    #     input_ids = torch.tensor(batch["input_ids"])
    #     bbox = torch.tensor(batch["bbox"])
    #     image = ImageList.from_tensors([torch.tensor(batch["image"])])
    #     att_mask = torch.tensor(batch["attention_mask"])
    #     token_type_ids = torch.tensor(batch["token_type_ids"])
    #     with torch.no_grad():
    #         y = self.model(input_ids, bbox, image, att_mask, token_type_ids)

    #     return {"class_index": y["logits"][0].argmax(axis=1).numpy()}
    
    async def __call__(self, starlette_request):
        image_payload_bytes = await starlette_request.body()
        read = self.reader.readtext(image_payload_bytes)
        pil_image = Image.open(BytesIO(image_payload_bytes))
        pil_image.save("/tmp/image.png")
        # read, convert and resize image
        image, size = load_image("/tmp/image.png")
        # tokenize_and_align bugs with 1-example datasets, so we add the parsed data twice.
        example = {
            "id": ["0", "1"],
            "bboxes": [[]],
            "tokens": [[]],
            "image": [image, image],
            "ner_tags": [[1]*len(read), [1]*len(read)]
        }
        for segment in read:
            # skip excerpts with low confidence 
            if segment[2] < 0.1:
                continue
            example["tokens"][0].append(segment[1])
            # bounding box first element is top left corner's x and y, bottom right is the third element.
            bbox = segment[0][0]+segment[0][2]
            example["bboxes"][0].append(normalize_bbox(bbox, size))

        example["bboxes"].append(example["bboxes"][0])
        example["tokens"].append(example["tokens"][0])
        inference_dataset =  Dataset.from_dict(example)
        remove_columns = inference_dataset.column_names
        inference_dataset = inference_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=1,
            load_from_cache_file=False,
        )
        # we let data_collator take care of the torch conversion
        batch = self.data_collator([inference_dataset[0]])
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        image = batch["image"]
        att_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        with torch.no_grad():
            y = self.model(input_ids, bbox, image, att_mask, token_type_ids)

        return {"class_index": y["logits"][0].argmax(axis=1).numpy()}

#%%
serve.start()
LayoutModel.deploy()
# %%
