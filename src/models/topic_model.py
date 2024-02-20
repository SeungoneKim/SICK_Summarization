from transformers import pipeline
import json
import numpy as np

class TopicModel:
  def __init__(self, path_label_json='/content/SICK_Summarization/src/data/Topic labels.json', confidence_threshold=0.1, top_k=2):
    self.model_name = "Recognai/zeroshot_selectra_medium"
    self.model = classifier = pipeline("zero-shot-classification", model=self.model_name) #22.5 M params
    with open(path_label_json, 'r') as file:
      self.labels = json.load(file)[:10] #TODO
    self.confidence_threshold = confidence_threshold
    self.top_k = top_k

  def predict(self, dialogue) -> list:
    out = []
    try:
      dialogue = ', '.join(dialogue)
      pred = self.model(dialogue, self.labels)
      labels = pred['labels']#[:self.top_k]
      scores = pred['scores']#[:self.top_k]
      for label, score in zip(labels, scores):
        if score > 1/len(labels):
          out.append(label)
    except:
      pass
    return out
