from bertopic import BERTopic


class TopicBERT:
    def __init__(self, confidence_threshold=0.1):
        self.model_name = "MaartenGr/BERTopic_Wikipedia"
        self.model = BERTopic.load(self.model_name)
        self.labels = self.model.get_topic_info()["Name"].to_list()
        self.confidence_threshold = confidence_threshold

    def predict(self, dialogue) -> list:
        label = []
        try:
            idx, p = self.model.transform(dialogue)
            idx, p = idx[0], p[0]
            if p > self.confidence_threshold:
                label = self.labels[idx].split("_")[1:]
        except:
            pass
        return label
