import json
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

# Initialize NLP model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to load datasets from JSON files
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load PIE dataset idioms and meanings from JSON
formal_idioms_data = load_dataset('path/to/formal_idioms.json')
static_idioms_data = load_dataset('path/to/static_idioms.json')

# Combine formal and static idioms into a single list
idioms_data = formal_idioms_data + static_idioms_data
idioms = [item['idiom'] for item in idioms_data]
meanings = [item['meaning'] for item in idioms_data]

# Function to generate embeddings
def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    output = model(**tokens)
    return output.last_hidden_state.mean(dim=1)

# Function to augment a single dialogue entry with an idiom
def augment_dialogue_entry(dialogue_entry, idioms, meanings):
    sentence = dialogue_entry['sentence']
    sentence_embedding = get_embedding(sentence)
    
    # Randomly select an idiom and its meaning
    idx = random.randint(0, len(idioms) - 1)
    selected_idiom, selected_meaning = idioms[idx], meanings[idx]
    idiom_embedding = get_embedding(selected_meaning)
    
    # Compute semantic similarity
    similarity = 1 - cosine(idiom_embedding.detach().numpy(), sentence_embedding.detach().numpy())
    
    # Decide whether to augment based on similarity threshold
    if similarity > 0.5:  # Threshold is arbitrary and can be adjusted
        augmented_sentence = sentence + f" This is like when people say, '{selected_idiom}'."
        dialogue_entry['sentence'] = augmented_sentence
    
    return dialogue_entry

# Load SAMSum dataset

json_path_train = "/content/drive/MyDrive/COMET_data/paracomet/dialogue/samsum/dialog_train_split5_collated.json"
json_path_test = "/content/drive/MyDrive/COMET_data/paracomet/dialogue/samsum/dialog_test_split5_collated.json"
json_path_validation = "/content/drive/MyDrive/COMET_data/paracomet/dialogue/samsum/dialog_validation_split5_collated.json"

# Augment dialogues
augmented_data = {}
for dialogue_id, dialogue_entries in samsum_data.items():
    augmented_entries = [augment_dialogue_entry(entry, idioms, meanings) for entry in dialogue_entries]
    augmented_data[dialogue_id] = augmented_entries

# Function to save augmented dataset
def save_augmented_data(augmented_data, file_path):
    with open(file_path, 'w') as file:
        json.dump(augmented_data, file, indent=4)

# Save the augmented SAMSum dataset
save_augmented_data(augmented_data, 'path/to/augmented_samsum_dataset.json')
