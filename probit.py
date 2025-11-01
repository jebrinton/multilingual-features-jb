import probing
import probe_features
import argparse

probing_args = argparse.Namespace(
    model_name="meta-llama/Llama-3.1-8B",
    batch_size=16,
    layer_num=16,
    seed=42,
    # language="Dutch-Alpino",
    language="English-PUD",
    # language="Ukrainian-ParlaMint",
    
)

pf_args = argparse.Namespace(
    concept_key="Tense",
    concept_value="Past",
    model_name="meta-llama/Llama-3.1-8B",
    seed=42,
)


probing.main(probing_args)

# probe_features.feature_selection(pf_args)

