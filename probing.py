import argparse
import glob
import logging
import os
from functools import partial

import joblib
import torch
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import HF_TOKEN
from src.probing.utils import extract_activations, concept_filter, get_available_languages, get_features_and_values
from src.probing.data import ProbingDataset, balance_dataset

from src.config import UD_BASE_FOLDER
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Constants
TRACER_KWARGS = {'scan': False, 'validate': False}
LOG_DIR = 'logs'

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'probing.txt'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def setup_model(model_name):
    """Initialize and return the language model."""
    return LanguageModel(model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN)

def get_ud_filepaths(language):
    """Get the filepaths for Universal Dependencies train and test files across all treebanks."""
    train_filepaths = []
    test_filepaths = []
    
    # Find all directories matching UD_{language} or UD_{language}-*
    if isinstance(UD_BASE_FOLDER, str):
        base_path = UD_BASE_FOLDER
    else:
        base_path = str(UD_BASE_FOLDER)
    
    for folder in os.listdir(base_path):
        if folder == f"UD_{language}" or folder.startswith(f"UD_{language}-"):
            ud_folder = os.path.join(base_path, folder)
            if os.path.isdir(ud_folder):
                # Glob for train files in this treebank
                train_files = glob.glob(os.path.join(ud_folder, "*-ud-train.conllu"))
                train_filepaths.extend(train_files)
                # Glob for test files in this treebank
                test_files = glob.glob(os.path.join(ud_folder, "*-ud-test.conllu"))
                test_filepaths.extend(test_files)
    
    return train_filepaths, test_filepaths

def prepare_datasets(train_filepaths, test_filepaths, concept_key, concept_value, seed):
    """Prepare and balance the training and test datasets."""
    filter_criterion = partial(concept_filter, concept_key=concept_key, concept_value=concept_value)
    train_dataset = ProbingDataset(train_filepaths, filter_criterion)
    test_dataset = ProbingDataset(test_filepaths, filter_criterion)

    print("Balancing training dataset...")
    train_dataset = balance_dataset(train_dataset, seed)
    print("Balancing test dataset...")
    test_dataset = balance_dataset(test_dataset, seed)

    return train_dataset, test_dataset

def get_best_classifier(train_activations, train_labels, seed):
    """Hyperparameter search for the logistic regression probe."""

    # get number of CPU
    num_cpu = int(os.environ.get("NSLOTS", 1))

    probe_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=seed,
            class_weight="balanced"
        ))
    ])

    param_grid = [{
        'model__C' : np.logspace(1, 1, 1),
        'model__penalty': ['l2'],
        'model__solver': ['saga'],
        'model__max_iter': [100]
    }]

    grid_search = GridSearchCV(
        probe_pipeline,
        param_grid, 
        cv=2,
        scoring='accuracy',
        n_jobs=num_cpu-2,
        verbose=2
    )
    
    grid_search.fit(train_activations, train_labels)
    best_classifier = grid_search.best_estimator_
    print(f"Best classifier: {best_classifier}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    print(f"Coefficients: {best_classifier.named_steps['model'].coef_}")
    print(f"Num non-zero coeffs: {np.count_nonzero(best_classifier.named_steps['model'].coef_)}")
    return best_classifier

def train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, seed):
    """Train a logistic regression probe and evaluate its performance."""
    print("Training logistic regression model...")
    classifier = get_best_classifier(train_activations, train_labels, seed)
    classifier.fit(train_activations, train_labels)

    train_accuracy = classifier.score(train_activations, train_labels)
    test_accuracy = classifier.score(test_activations, test_labels)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return classifier

def process_language(args, language):
    """Process a single language for probing."""
    print(f"\nProcessing language: {language}")
    logging.info(f"Processing language: {language}")
    
    model = setup_model(args.model_name)
    train_filepaths, test_filepaths = get_ud_filepaths(language)

    if not train_filepaths or not test_filepaths:
        print(f"Skipping {language}: Missing train or test files")
        logging.warning(f"Skipping {language}: Missing train or test files")
        return

    # Get features from all train files
    features = get_features_and_values(train_filepaths)

    output_dir = f"outputs/probing/probes/{'llama' if 'llama' in args.model_name else 'aya'}"
    for concept_key, values in features.items():
        for concept_value in values:
            # Only process Number concept for testin
            if concept_key != "Number":
                continue

            print(f"\nProcessing {language} - {concept_key}: {concept_value}")
            logging.info(f"Processing {language} - {concept_key}: {concept_value}")

            model_filename = f"{language}_{concept_key}_{concept_value}.joblib"
            model_path = os.path.join(output_dir, model_filename)
            
            if os.path.exists(model_path):
                print(f"Probe already exists. Skipping.")
                logging.info(f"Probe already exists for {language} - {concept_key}: {concept_value}. Skipping.")
                continue

            train_dataset, test_dataset = prepare_datasets(train_filepaths, test_filepaths, concept_key, concept_value, args.seed)

            MIN_SAMPLES = 512
            if train_dataset is None or len(train_dataset) < MIN_SAMPLES or test_dataset is None:
                logging.warning(f"Skipping {language} - {concept_key}: {concept_value}: less than {MIN_SAMPLES} samples")
                continue

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            print("Extracting activations...")
            train_activations, train_labels = extract_activations(model, train_dataloader, args.layer_num)
            test_activations, test_labels = extract_activations(model, test_dataloader, args.layer_num)

            classifier = train_and_evaluate_probe(train_activations, train_labels, test_activations, test_labels, args.seed)

            os.makedirs(output_dir, exist_ok=True)
            joblib.dump(classifier, model_path)
            print(f"Saved trained model to {model_path}")
            logging.info(f"Saved trained model to {model_path}")

def main(args):
    if args.language:
        languages = [args.language]
        # Check if any directory matching UD_{language}* exists
        if isinstance(UD_BASE_FOLDER, str):
            base_path = UD_BASE_FOLDER
        else:
            base_path = str(UD_BASE_FOLDER)
        matching_dirs = [d for d in os.listdir(base_path) if d == f"UD_{args.language}" or d.startswith(f"UD_{args.language}-")]
        if not matching_dirs:
            print(f"Error: Language '{args.language}' not found in Universal Dependencies folder.")
            logging.error(f"Language '{args.language}' not found in Universal Dependencies folder.")
            return
    else:
        languages = get_available_languages(UD_BASE_FOLDER)
    
    for language in languages:
        process_language(args, language)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probing script for language models")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the language model to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading")
    parser.add_argument("--layer_num", type=int, default=16, help="Layer number to extract activations from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--language", type=str, help="Specific language to process (optional)")
    args = parser.parse_args()

    main(args)