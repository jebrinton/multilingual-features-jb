from tqdm import tqdm
from src.probing.utils import extract_word_activations
# from src.probing.data import WordProbingDataset
from src.probing.utils import concept_filter
from nnsight import LanguageModel
from torch.utils.data import DataLoader, Subset
from functools import partial
import torch
import glob
from src.config import UD_BASE_FOLDER
import os
from transformers import AutoModel, AutoTokenizer
import pyconll
import joblib
import numpy as np

from probing import get_best_classifier, train_and_evaluate_probe

DATA_DIR = "/projectnb/mcnet/jbrin/multilingual-features-jb/data/processed_sentences"
os.makedirs(DATA_DIR, exist_ok=True)

from torch.utils.data import Dataset

class WordProbingCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of tuples: [ (word_list_1, label_list_1), (word_list_2, label_list_2), ... ]
        
        # 1. Unzip the batch
        word_form_lists = [item[0] for item in batch]
        word_label_lists = [item[1] for item in batch] # This is a list of lists

        # 2. Tokenize the word lists
        tokenized_batch = self.tokenizer(
            word_form_lists, 
            is_split_into_words=True,  # This is the key!
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # 3. Return the tokenized batch and the un-stacked labels
        # The loop will get (tokenized_batch, word_label_lists)
        return tokenized_batch, word_label_lists

class WordProbingDataset(Dataset):
    def __init__(self, processed_sentences, concept_key, concept_value):
        """
        Args:
            processed_sentences (list[list[dict]]): Parsed sentences data.
            concept_key (str): The feature to probe (e.g., "Number").
            concept_value (str): The value to treat as '1' (e.g., "Plur").
        """
        self.word_forms = []
        self.word_labels = []

        for proc_sentence in processed_sentences:
            if not proc_sentence: # skip empty sentences
                continue
            
            word_forms = []
            labels = []

            for word_dict in proc_sentence:
                word_form = word_dict.get("form")
                
                # This is the fix:
                if word_form is None:
                    word_form = "" # Replace None with an empty string
                    
                word_forms.append(word_form)
                
                feats = word_dict.get("feats", {})
                label = 1 if concept_value in feats.get(concept_key, set()) else 0
                labels.append(label)
            
            self.word_forms.append(word_forms)
            self.word_labels.append(labels)

    def __len__(self):
        if (len(self.word_forms) != len(self.word_labels)):
            raise ValueError("word_forms and word_labels must have the same length")
        return len(self.word_forms)

    def __getitem__(self, idx):
        return self.word_forms[idx], self.word_labels[idx]

def conllu_to_processed_sentences(conll_filepaths):
    """
    Convert a list of conllu filepaths to a list of processed sentences.
    Args:
        conll_filepaths (list[str]): List of conllu filepaths.
    Returns:
        list[list[dict]]: List of processed sentences. List of sentences, each sentence is a list of word dictionaries.
    """
    processed_sentences = []
    for conll_file in tqdm(conll_filepaths, desc=f"Parsing conllu files"):
        for sentence in pyconll.iter_from_file(conll_file):
            processed_ids = set()
            processed_sentence = []
            for i, word in enumerate(sentence):
                # already processed via MWT
                if word.id in processed_ids:
                    continue

                word_dict = {"id": word.id, "form": word.form, "feats": word.feats}

                if word.is_multiword():
                    # kinda some weird logic here
                    # first we get a tuple of the MWT span as ints,
                    # the convert back to strings to index into the sentence
                    # this is important! 
                    # in conlllu, sentence[i] ≠ sentence[str(i)] due to multiword tokens
                    span = tuple(int(x) for x in word.id.split("-"))
                    start, end = span
                    span_ids = [str(i) for i in range(start, end + 1)]
                    for id in span_ids:
                        word_dict["feats"].update(sentence[id].feats) # possible TODO: ensure that MWT feats can have multiple values
                        processed_ids.add(id)
                processed_sentence.append(word_dict)

            processed_sentences.append(processed_sentence)
    return processed_sentences

def get_processed_sentences(language, split="train"):
    """
    Get the processed sentences for a given language.
    Args:
        language (str): Language code.
        split (str): Split to get sentences for.
    Returns:
        list[list[dict]]: List of processed sentences.
    """
    cache_filename = f"{language}_{split}_processed.joblib"
    cache_path = os.path.join(DATA_DIR, cache_filename)

    # 1. Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading {language}:{split} sentences from cache...")
        processed_sentences = joblib.load(cache_path)
        return processed_sentences

    # 2. If cache doesn't exist, run your parsing code
    print(f"Parsing {language}:{split} from .conllu files (this may take a while)...")
    conll_filepaths = glob.glob(os.path.join(UD_BASE_FOLDER, f"UD_{language}*", f"*-ud-{split}.conllu"))
    processed_sentences = conllu_to_processed_sentences(conll_filepaths)
    joblib.dump(processed_sentences, cache_path)
    return processed_sentences

def extract_word_activations(hf_model, dataloader, layer_num):
    all_word_activations = []
    all_word_labels = []
    
    for tokenized_batch, batch_word_labels in tqdm(dataloader, desc="Extracting activations"):
        batch_word_ids = [tokenized_batch.word_ids(i) for i in range(len(batch_word_labels))]

        tokenized_batch = {k: v.to(device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            outputs = hf_model(**tokenized_batch, output_hidden_states=True)
            
        acts = outputs.hidden_states[layer_num] # [batch_size, seq_len, hidden_dim]

        # 4. Loop, align, and pool
        for i in range(len(batch_word_labels)): # Loop over sentences
            sentence_acts = acts[i]
            sentence_word_ids = batch_word_ids[i]
            sentence_labels = batch_word_labels[i]
            
            num_words_in_sentence = len(sentence_labels)

            for word_index in range(num_words_in_sentence): # Loop over words
                
                # Find all tokens for this word
                token_span = [j for j, wid in enumerate(sentence_word_ids) 
                                if wid == word_index]
                
                if not token_span:
                    continue # Word was truncated or lost
                    
                # Get all token activations for this word
                token_activations = sentence_acts[token_span]
                
                # Mean-pool the tokens
                word_activation = token_activations.mean(dim=0)
                
                all_word_activations.append(word_activation.float().cpu().numpy())
                all_word_labels.append(sentence_labels[word_index])
    
    # Stack all word activations into a 2D array [num_total_words, hidden_dim]
    return np.vstack(all_word_activations), np.array(all_word_labels)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
nnsight_model = LanguageModel(model_name, torch_dtype=torch.float16, device_map="auto")
my_hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# processed_sentences = get_processed_sentences("Spanish", "train")
# dataset = WordProbingDataset(processed_sentences, "Number", "Plur")
# subset_indices = range(64)
# dataset = Subset(dataset, subset_indices)

# dataloader = DataLoader(
#     dataset, 
#     batch_size=16, 
#     shuffle=False, 
#     collate_fn=WordProbingCollate(tokenizer)
# )


# word_acts, word_labels = extract_word_activations(my_hf_model, dataloader, layer_num)

# # save word_acts and word_labels to a file
# np.savez("word_acts_and_labels.npz", word_acts=word_acts, word_labels=word_labels)

# # load word_acts and word_labels from a file
# word_acts, word_labels = np.load("word_acts_and_labels.npz")

# classifier = train_and_evaluate_probe(word_acts, word_labels, word_acts, word_labels, 42)

# classifier.fit(train_activations, train_labels)

# train_accuracy = classifier.score(word_acts, word_labels)
# test_accuracy = classifier.score(word_acts, word_labels)

# print(f"Train Accuracy: {train_accuracy:.2f}")
# print(f"Test Accuracy: {test_accuracy:.2f}")

CONCEPTS_VALUES = {
    "Number": ["Sing", "Plur"],
    "Tense": ["Past", "Pres"],
    "Gender": ["Masc", "Fem"],
}
LANGUAGES = ["English", "French", "Turkish"]

layer_num = 16
for concept, values in CONCEPTS_VALUES.items():
    for value in values:
        for language in LANGUAGES:
            train_sentences = get_processed_sentences(language, "train")
            test_sentences = get_processed_sentences(language, "test")
            train_dataset = WordProbingDataset(train_sentences, concept, value)
            test_dataset = WordProbingDataset(test_sentences, concept, value)
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=WordProbingCollate(tokenizer))
            test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=WordProbingCollate(tokenizer))
            
            train_word_acts, train_word_labels = extract_word_activations(my_hf_model, train_dataloader, layer_num)
            test_word_acts, test_word_labels = extract_word_activations(my_hf_model, test_dataloader, layer_num)
            
            print(f"For {language} {concept} {value}:")
            classifier = train_and_evaluate_probe(train_word_acts, train_word_labels, test_word_acts, test_word_labels, 42)


