import pytorch_lightning as L
from config import RANDOM_SEED, AVAIL_GPUS, EXPERIMENTS, get_parameters
from graphcpp.lightning import GraphCPPModule, GraphCPPDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from datetime import datetime
import argparse
import json
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def run_one_training(experiment_name, dataset, **model_kwargs):
    mlf_logger = CSVLogger(save_dir="logs", name=experiment_name)

    # Create datamodule for the current fold
    module = GraphCPPDataModule(folder=dataset, fp_type=model_kwargs['fingerprint_type'])

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        # callbacks=[EarlyStopping(monitor="val_mcc", mode="max", patience=20), ModelCheckpoint(monitor="val_mcc", mode="max")],
        callbacks=[ModelCheckpoint(monitor="val_mcc", mode="max")],
        accelerator='cuda' if AVAIL_GPUS > 0 else 'cpu',
        devices=AVAIL_GPUS if AVAIL_GPUS > 0 else 'auto',
        max_epochs=100,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        logger=mlf_logger
    )

    model = GraphCPPModule(**model_kwargs)
    trainer.fit(model, train_dataloaders=module.train_dataloader(), val_dataloaders=module.val_dataloader())
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = GraphCPPModule.load_from_checkpoint(best_model_path)

    # Validate and test the model
    trainer.validate(model, dataloaders=module.val_dataloader(), verbose=True)
    test_results = trainer.test(model, dataloaders=module.test_dataloader(), verbose=True)
    return test_results[0], best_model_path, count_parameters(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning models")
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    start_time = datetime.now()
    parser.add_argument('--experiment_name', type=str, help='Experiment name (defaults to --config if given)', default=None)
    # dataset path folder
    parser.add_argument('--dataset', type=str, help='Dataset folder', default='dataset')
    parser.add_argument('--config', type=str, choices=sorted(EXPERIMENTS), help='Named architecture from config.EXPERIMENTS (default: BEST_PARAMETERS)', default=None)
    parser.add_argument('--pos_weight', type=float, help='BCE positive class weight for imbalanced training sets (see prepare_merged_dataset.py)', default=None)
    parser.add_argument('--results_dir', type=str, help='Where to write the result json (use a separate dir per dataset)', default='results')
    args = parser.parse_args()
    experiment_name = args.experiment_name or args.config or f"{start_time}"
    params = get_parameters(args.config, pos_weight=args.pos_weight)

    L.seed_everything(args.seed)
    print(f"Seed: {args.seed}")
    print(f"Experiment: {experiment_name}")
    print(f"layer_type: {params['layer_type']}, mp_layers: {params['mp_layers']}, hidden_channels: {params['hidden_channels']}")

    print(f"Start time: {start_time}")

    test_results, best_model_path, n_params = run_one_training(experiment_name, args.dataset, **params)
    print(f"Parameter count: {n_params}")
    print(f"Test MCC: {test_results['test_mcc']}")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f'{experiment_name}.json'), 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'config': args.config,
            'seed': args.seed,
            'dataset': args.dataset,
            'parameters': params,
            'parameter_count': n_params,
            'best_model_path': best_model_path,
            'test_metrics': test_results,
        }, f, indent=2)
