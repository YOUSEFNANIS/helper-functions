import gc
import torch
from typing import Any
from dataclasses import dataclass

def apply_masking(text, mask_rate=0.07):
    gc.enable()
    mask = torch.rand(text.shape) > mask_rate
    text = text * mask
    mask = None
    gc.collect()
    return text

def describe_model(model):
    print(f'Memory used by model: {round(model.get_memory_footprint()/1024/1024/1024, 2)} GB')
    print(f'total number of parameters is {sum(p.numel() for p in model.parameters())}')

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        gc.enable()
        input_features = [{"input_features": apply_masking(feature["input_features"].squeeze(0))} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"].squeeze(0)} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        gc.collect()
        return batch