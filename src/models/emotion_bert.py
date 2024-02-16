from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np


class EmotionBERT:
    def __init__(
        self,
        path_load: str = None,
        path_save: str = None,
        tokenizer_name: str = "bert-base-uncased",
    ) -> None:
        self.path_load = path_load
        self.path_save = path_save
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path_load, use_safetensors=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.labels = [
            "anger",
            "anticipation",
            "disgust",
            "fear",
            "joy",
            "love",
            "optimism",
            "pessimism",
            "sadness",
            "surprise",
            "trust",
        ]

    # Input it's a dialogue: list[str]
    def predict(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors="pt")
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        outputs = self.model(**encoding)
        logits = outputs.logits
        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())

        if any(probs >= 0.4):
            probs = probs.detach().numpy()
            probs = probs / sum(probs)
            percentile = 10
            quantile = np.percentile(probs, 100 - percentile)
            mask = probs >= quantile
            return [self.labels[i] for i in range(len(self.labels)) if mask[i]]
        else:
            return []
