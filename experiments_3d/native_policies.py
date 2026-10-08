"""Load verified authors' neural definitions without their CLI training jobs.

Only class definitions are compiled, byte-for-byte from the pinned source AST.
Task encoders adapt feature dimensions; network bodies remain upstream code.
This module alone does not constitute trained baseline evaluation evidence.
"""
from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments_3d.data_upgrade import read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "Data/external/multicomponent_3d_public_sources_20260928"


def upstream_classes(method: str, filename: str, names: tuple[str, ...], extra=None):
    directory = SOURCES / method
    manifest = read_json(directory / "source_manifest.json")
    record = next(r for r in manifest["files"] if r["path"] == filename)
    path = directory / filename
    if sha256(path) != record["sha256"]:
        raise ValueError("upstream neural source hash changed")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("upstream neural class set changed")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"__name__": f"paper10_upstream_{method}", "torch": torch, "nn": nn, "F": F, "np": np, "random": random,
                 "device": torch.device("cpu"), **(extra or {})}
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names}), manifest


def pyg():
    runtime = ROOT / "Data/runtime/multicomponent_3d_20260928"
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
    return {"Data": Data, "DataLoader": DataLoader, "GCNConv": GCNConv, "GATv2Conv": GATv2Conv, "SAGEConv": SAGEConv}


class AuthorsNLNSRepair(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        source, self.source_manifest = upstream_classes("nlns", "actor.py", ("Encoder", "Attention", "Pointer", "VrpActorModel"))
        self.actor = source.VrpActorModel(torch.device("cpu"), hidden_size=hidden_dim)
        self.actor.all_embed = source.Encoder(8, hidden_dim)
        self.actor.origin_embed = source.Encoder(8, hidden_dim)
        for encoder in (self.actor.all_embed, self.actor.origin_embed):
            for parameter in encoder.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.feature_dim, self.hidden_dim = 8, hidden_dim

    def forward(self, features, origin_features=None):
        if features.ndim != 2 or features.shape[1] != 8 or not len(features):
            raise ValueError("NLNS service features must have shape [candidate, 8]")
        origin = features.mean(0) if origin_features is None else origin_features.reshape(8)
        # Native forward squeezes the batch dimension for B=1. Duplicate the
        # identical state to B=2 and return only sample 0, preserving its math.
        batch = features.unsqueeze(0).repeat(2, 1, 1)
        origins = origin.unsqueeze(0).repeat(2, 1)
        static_index, dynamic_index = [1, 2, 3, 7], [4, 0, 5, 6]
        return self.actor(batch[:, :, static_index].clone(), batch[:, :, dynamic_index], origins[:, static_index],
                          origins[:, dynamic_index], torch.ones(2, len(features), dtype=torch.bool))[0]


class AuthorsNLNSCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        source, self.source_manifest = upstream_classes("nlns", "critic.py", ("Encoder", "VrpCriticModel"))
        self.critic = source.VrpCriticModel(hidden_dim)
        self.critic.encoder, self.critic.encoder_2 = source.Encoder(8, hidden_dim), source.Encoder(8, hidden_dim)
        for encoder in (self.critic.encoder, self.critic.encoder_2):
            nn.init.xavier_uniform_(encoder.conv.weight)

    def forward(self, features):
        batch = features.unsqueeze(0).repeat(2, 1, 1)
        return self.critic(batch[:, :, [1, 2, 3, 7]], batch[:, :, [4, 0, 5, 6]])[0, 0]


def authors_dac(hidden_dim=64, destroy_actions=12):
    source, manifest = upstream_classes("dac_alns", "dac-alns-cvrp.py", ("GraphEncoder", "DACAgent"), pyg())
    class InspectionDAC(source.DACAgent):
        # Upstream fixes exploration to two CVRP actions. Adapt its output
        # boundary to the same inspection operator/ratio library as ALNS/GRLOS;
        # the two-stage update, shared critic and categorical sampler are kept.
        def act_destroy(self, state, eps=.1):
            return self._sample_with_eps(self.actor_destroy(state), eps, self.n_destroy)

    policy = InspectionDAC(enc_dim=hidden_dim, hid=hidden_dim)
    if destroy_actions < 1:
        raise ValueError("positive task destroy action count required")
    head = policy.actor_destroy[-1]
    policy.actor_destroy[-1] = nn.Linear(head.in_features, destroy_actions)
    policy.n_destroy = destroy_actions
    policy.opt_ad = torch.optim.Adam(policy.actor_destroy.parameters(), lr=1e-3)
    # Eight task features plus directly observed selected-candidate membership.
    policy.encoder = source.GraphEncoder(in_dim=9, hid=hidden_dim, out_dim=hidden_dim, use_attention=True)
    policy.opt_enc = torch.optim.Adam(policy.encoder.parameters(), lr=1e-3)
    return policy, manifest


class ServiceFeatureProcessor:
    def get_feature_sizes(self):
        return 9, 1, 4

    def get_states_representations(self, states, **kwargs):
        return states


def authors_grlos(hidden_dim=32):
    extra = {**pyg(), "Linear": nn.Linear, "kaiming_uniform_": nn.init.kaiming_uniform_,
             "xavier_uniform_": nn.init.xavier_uniform_, "xavier_normal_": nn.init.xavier_normal_}
    source, manifest = upstream_classes("grlos", "alns/rl/dqn/q_net.py", ("QNet", "NStepQNet"), extra)
    library = SimpleNamespace(unique_ops_destroy=[f"destroy_{i}" for i in range(12)], unique_ops_repair=[f"repair_{i}" for i in range(3)])
    network = source.NStepQNet(ServiceFeatureProcessor(), {"arch": "gat", "num_hidden_layers": 3, "lf_dim": hidden_dim}, library, "inspection")
    return network, manifest


def smoke():
    torch.manual_seed(23)
    features = torch.rand(5, 8)
    features[:, 0] = .2
    actor, critic = AuthorsNLNSRepair(16), AuthorsNLNSCritic(16)
    logits, value = actor(features), critic(features)
    loss = -(torch.log_softmax(logits, 0)[0]) + value.square()
    loss.backward()
    Data = pyg()["Data"]
    graph = Data(x=torch.cat((features,torch.zeros(5,1)),dim=1), edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]]),
                 edge_attr=torch.ones(8, 1), edge_weight=torch.ones(8), global_feats=torch.zeros(4), state_num_nodes=torch.tensor([5]))
    dac, dac_source = authors_dac(16)
    state = dac.build_state(graph, .2, 0)
    d, r = dac.act_destroy(state, eps=0), dac.act_repair(state, eps=0)
    d_trans = {"states": [state], "actions": [d], "rewards": [.1], "next_states": [state.detach()], "dones": [True]}
    # Rebuild graph state for the second stage, as in the original update path.
    r_state = dac.build_state(graph, .2, d)
    r_trans = {"states": [r_state], "actions": [r], "rewards": [.1], "next_states": [r_state.detach()], "dones": [True]}
    dac.update(d_trans, r_trans)
    grlos, grlos_source = authors_grlos(16)
    qd, qr = grlos(0, [graph]), grlos(1, [graph])
    (qd.square().mean() + qr.square().mean()).backward()
    if not all(torch.isfinite(t).all() for t in (logits, value, qd, qr)):
        raise ValueError("nonfinite native neural output")
    return {"scope": "native network forward/backward interface only; target-task training and search fidelity pending",
        "nlns": {"commit": actor.source_manifest["commit"], "actor_output_shape": list(logits.shape), "critic_scalar": float(value.detach())},
        "dac_alns": {"commit": dac_source["commit"], "native_update_completed": True, "destroy_actions": dac.n_destroy, "repair_actions": 3},
        "grlos": {"commit": grlos_source["commit"], "destroy_q_shape": list(qd.shape), "repair_q_shape": list(qr.shape)}}


if __name__ == "__main__":
    import argparse
    import json
    from experiments_3d.data_upgrade import write_once
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = smoke()
    write_once(args.output, report)
    print(json.dumps(report))
