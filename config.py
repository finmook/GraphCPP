import torch

# Random seed
RANDOM_SEED = 42

# Cuda arguments
AVAIL_GPUS = min(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
BATCH_SIZE = 512

BEST_PARAMETERS = {
    'act': 'prelu',
    'conv_aggr': 'sum',
    'conv_dropout': 0.05,
    'has_bn': False,
    'has_l2norm': False,
    'layer_fingerprints': 1,
    'fingerprint_type': 'topological',
    'hidden_channels': 128,
    'layer_type': 'sageconv',
    'layers_pre_mp': 1,
    'mp_layers': 4,
    'layers_post_mp': 1,
    'learning_rate': 0.001,
    'pooling': 'mean',
    'stage_type': 'stack',
    'weight_decay': 'cosine'
}

# Progress Report #3 architecture comparison.
# Each entry only overrides the GNN layer settings of BEST_PARAMETERS (GraphSAGE-4, the PR2 reference);
# everything else (data, split, seed, optimizer, scheduler, pooling, fingerprint branch, threshold) is shared.
EXPERIMENTS = {
    'graphcpp_sage2': {'layer_type': 'sageconv', 'mp_layers': 2},
    'graphcpp_sage4': {'layer_type': 'sageconv', 'mp_layers': 4},
    'graphcpp_sage6': {'layer_type': 'sageconv', 'mp_layers': 6},
    'graphcpp_transformer2': {'layer_type': 'transformerconv', 'mp_layers': 2, 'n_heads': 4},
    'graphcpp_transformer4': {'layer_type': 'transformerconv', 'mp_layers': 4, 'n_heads': 4},
}

def get_parameters(config_name=None):
    params = dict(BEST_PARAMETERS)
    if config_name is not None:
        params.update(EXPERIMENTS[config_name])
    return params
