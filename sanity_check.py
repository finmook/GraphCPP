# Pre-training sanity check for the Progress Report #3 architectures.
# Builds every config in config.EXPERIMENTS, passes one real CPP1708 batch through it,
# computes the BCE loss and a backward pass, and prints a summary table.

import argparse
import torch
import pytorch_lightning as L
from torch_geometric.loader import DataLoader
from config import BATCH_SIZE, EXPERIMENTS, get_parameters
from graphcpp.dataset import CPPDataset
from graphcpp.lightning import GraphCPPModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset/raw/2024-07-14-15-40-08')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    params = get_parameters()
    dataset = CPPDataset(root=args.dataset, _split=args.split, fp_type=params['fingerprint_type'])
    batch = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False))).to(device)
    print(f"Real batch ({args.split}): graphs={batch.num_graphs}, x={tuple(batch.x.shape)}, "
          f"edge_index={tuple(batch.edge_index.shape)}, edge_attr={tuple(batch.edge_attr.shape)}, fp={tuple(batch.fp.shape)}")

    rows = []
    for name in EXPERIMENTS:
        params = get_parameters(name)
        status = 'OK'
        n_params = None
        try:
            L.seed_everything(args.seed, workers=False)
            module = GraphCPPModule(**params).to(device)
            n_params = sum(p.numel() for p in module.parameters())
            preds, labels, _ = module(batch.clone())
            assert preds.shape == (batch.num_graphs, 1), f"logit shape {tuple(preds.shape)}"
            loss = module.loss_module(preds.squeeze(1), labels.float())
            assert torch.isfinite(loss), "non-finite loss"
            loss.backward()
            status = f"OK (logits {tuple(preds.shape)}, BCE {loss.item():.4f})"
        except Exception as e:
            status = f"FAILED: {type(e).__name__}: {e}"
        rows.append((name, params['layer_type'], params['mp_layers'], params['hidden_channels'], n_params, status))

    print()
    print(f"{'Model':<24}{'layer_type':<17}{'mp_layers':>10}{'hidden':>8}{'params':>10}  forward-pass status")
    for name, layer_type, mp_layers, hidden, n_params, status in rows:
        print(f"{name:<24}{layer_type:<17}{mp_layers:>10}{hidden:>8}{str(n_params):>10}  {status}")

    if any(not r[-1].startswith('OK') for r in rows):
        raise SystemExit(1)
    print("\nALL SANITY CHECKS PASSED")
