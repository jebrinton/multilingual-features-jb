from src.probing.utils import extract_word_activations
from src.probing.data import WordProbingDataset
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



# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = AutoModel.from_pretrained(model_name).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token

conll_filepaths = glob.glob(os.path.join(UD_BASE_FOLDER, "UD_Spanish*", "*-ud-train.conllu"), recursive=True)
filter_criterion = partial(concept_filter, concept_key="Number", concept_value="Plur")

conll_file = conll_filepaths[0]

end_on_this_sentence = False
processed_sentences = []
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

# TODO: put processed_sentences into a dataset
print("Need to process this into a dataset")
exit()

dataset = WordProbingDataset(conll_filepaths, filter_criterion)
subset_indices = range(64)
dataset = Subset(dataset, subset_indices)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

layer_num = 16
for batch in dataloader:
    text_batch = batch["sentence"] # list of sentences, length batch_size
    labels = batch["label"] # wait labels are still being SENTENCE POOLED major TODO: change this
    print(labels, labels.shape)

    inputs = tokenizer(text_batch, return_tensors="pt", padding=True)

    print("word_ids:", inputs.word_ids())
    print(text_batch)

    # 2. Get word_ids for alignment
    # Replaces `inputs.word_ids()`
    # This creates a list of lists, one for each sentence in the batch
    batch_word_ids = [inputs.word_ids(i) for i in range(len(text_batch))]

    # Move tokenized inputs to the same device as the model
    inputs = inputs.to(model.device)

    # 3. Run the forward pass to get hidden states
    # Replaces `with model.trace(...)`
    # We use torch.no_grad() for maximum efficiency
    with torch.no_grad():
        model_output = model(
            **inputs,
            output_hidden_states=True # This is the key!
        )

    # 4. Extract the correct layer's activations
    # Replaces `acts = model.model.layers[layer_num].output[0].save()`
    
    # model_output.hidden_states is a tuple of all hidden states
    # [0] = input embeddings
    # [1] = output of layer 0
    # [2] = output of layer 1
    # ...
    # [17] = output of layer 16
    
    # So, to get the output of layer `layer_num`, we access index `layer_num + 1`
    acts = model_output.hidden_states[layer_num + 1] # (batch_size, sequence_length, hidden_dim)
    
    # Print word_ids for the first sentence in the batch
    for i in range(len(text_batch)):
        if i != 0:
            continue
        print("word_ids (sentence):", batch_word_ids[i])
        print("words (sentence):", text_batch[i])
        print("labels (sentence):", labels[i])
        print("--------------------------------")

    # Now you would implement the alignment logic here, looping through the batch
    # and using 'acts[i]', 'batch_word_ids[i]', and 'labels[i]'

