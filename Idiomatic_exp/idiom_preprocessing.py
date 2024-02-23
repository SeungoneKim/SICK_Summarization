# Step 1: Loading and Parsing Idioms and Their Meanings
# First, let's write functions to load idioms and their meanings. For the meanings, we'll use BeautifulSoup to parse the HTML-like content and extract the text.

from transformers import BertTokenizer, BertModel
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
import json
from bs4 import BeautifulSoup

def load_idioms(file_path):
    with open(file_path, 'r') as file:
        idioms = [line.strip() for line in file.readlines()]
    return idioms

def parse_meaning_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = soup.find_all(text=True)
    return ' '.join(texts)

def load_meanings(file_path):
    meanings = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_html = parse_meaning_html(line)
            meanings.append(parsed_html)
    return meanings




# Step 3: Finding Closest Idiom and Translating
# We'll keep these functions mostly unchanged but make sure they work with the updated data structures.

from sentence_transformers import SentenceTransformer, util

# Load a sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_closest_idiom(sentence, idiom_dict, min_similarity=0.5):
    """
    Finds the closest idiom in the sentence based on semantic similarity.

    Parameters:
    - sentence (str): The input sentence to search for idioms.
    - idiom_dict (dict): A dictionary of idioms and their meanings.
    - min_similarity (float): The minimum similarity score to consider a match.

    Returns:
    - (str, str): The closest idiom and its meaning, or (None, None) if no suitable idiom is found.
    """

    sentence_embedding = model.encode([sentence], convert_to_tensor=True)
    highest_similarity = 0
    closest_idiom = None

    for idiom in idiom_dict.keys():
        idiom_embedding = model.encode([idiom], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(sentence_embedding, idiom_embedding)[0][0].item()

        if similarity > highest_similarity and similarity >= min_similarity:
            highest_similarity = similarity
            closest_idiom = idiom

    if closest_idiom:
        return closest_idiom, idiom_dict[closest_idiom]
    else:
        return None, None

def translate_sentence(sentence, idioms, meanings):
    closest_idiom = find_closest_idiom(sentence, idioms)
    if closest_idiom:
        idiom_index = idioms.index(closest_idiom)
        meaning = meanings[idiom_index]
        translated_sentence = sentence.replace(closest_idiom, meaning)
        return translated_sentence
    return sentence



# Load Spacy model for syntactic analysis
nlp = spacy.load("en_core_web_sm")

# Load a sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def translate_sentence(sentence, idioms,meanings):
    # Step 1: Idiom Replacement
    closest_idiom = find_closest_idiom(sentence, idioms)
    if closest_idiom:
        idiom_index = idioms.index(closest_idiom)
        meaning = meanings[idiom_index]
        translated_sentence = sentence.replace(closest_idiom, meaning)
        # Step 2: Syntactic Adjustment
        doc = nlp(translated_sentence)
        adjusted_sentence = syntactic_adjustment(doc)

        # Step 3: Semantic Validation
        original_embedding = model.encode([translated_sentence])
        adjusted_embedding = model.encode([adjusted_sentence])
        similarity = util.pytorch_cos_sim(original_embedding, adjusted_embedding)

        # If similarity is below a threshold, consider reverting or adjusting
        if similarity < 0.7:  # Example threshold
            # Consider alternative adjustments or revert to the original
            print("Semantic validation failed. Considering alternatives.")

        return adjusted_sentence
    return sentence



from transformers import pipeline

def syntactic_adjustment(sentence):
    # Initialize the text generation pipeline with GPT-3 or a similar powerful model
    generator = pipeline('text-generation', model='gpt-3', framework='pt')

    # Construct the prompt to instruct the model to correct the sentence
    prompt = f"Rewrite the following sentence for clarity and grammatical correctness: \"{sentence}\""

    # Generate the corrected sentence using the model
    corrected_sentences = generator(prompt, max_length=512, num_return_sequences=1)

    # Extract the first (and most likely the best) corrected sentence
    corrected_sentence = corrected_sentences[0]['generated_text'].split(prompt)[-1].strip()

    return corrected_sentence


# Example usage
sentence = "He spilled the beans during the meeting."
idiom_dict = {"spilled the beans": "revealed the secret"}
translated_sentence = translate_sentence(sentence, idiom_dict)
print(translated_sentence)

# Step 4: Handling the SAMSum Dataset (Continued)
# The function translate_samsum_dataset processes the SAMSum dataset by extracting each sentence, finding the closest idiom, translating the sentence by replacing the idiom with its meaning, and collecting the translated sentences.


def translate_samsum_dataset(samsum_file, output_file):
    with open(samsum_file, 'r') as file:
        samsum_data = json.load(file)
    
    translated_data = {}

    for dialog_id, dialog_data in samsum_data.items():
        translated_dialog = {}
        for message_id, message_data in dialog_data.items():
            sentence = message_data['sentence']
            # Translate using formal idioms first
            translated = translate_sentence(sentence, formal_idioms, formal_meanings)
            if translated == sentence:
                # If not found, try with static idioms
                translated = translate_sentence(sentence, static_idioms, static_meanings)
            translated_dialog[message_id] = message_data.copy()
            translated_dialog[message_id]['sentence'] = translated
        translated_data[dialog_id] = translated_dialog

    with open(output_file, 'w') as file:
        json.dump(translated_data, file, indent=4)
# Load idioms and their meanings
formal_idioms = load_idioms('formal_idioms.txt')
formal_meanings = load_meanings('formal_idioms_meanings.txt')
static_idioms = load_idioms('static_idioms.txt')
static_meanings = load_meanings('static_idioms_meanings.txt')


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
