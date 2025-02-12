import torch
import wandb
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf


def unnormalize_dict(normalized_dict, sep="/"):
    result = {}
    for key, value in normalized_dict.items():
        keys = key.split(sep)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result

def wandb_load_artifact(api, artifact_name):
    artifact = api.artifact(artifact_name)
    run = artifact.logged_by()
    config = run.config
    config = unnormalize_dict(config)
    config = OmegaConf.create(config)

    return config, artifact

def wandb_model_load(api, artifact_name):
    config, artifact = wandb_load_artifact(api, artifact_name)

    dir_name = artifact.download(root=Path("/tmp/wandb_downloads"))
    state_dict = torch.load(str(Path(dir_name) / "best_model.pth"))
    return config, state_dict


class MLP(nn.Module):
    def __init__(self, input_size, output_dim, num_units, num_layers, activation, final_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, num_units), activation])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_units, num_units))
            self.layers.append(activation)
        self.layers.append(nn.Linear(num_units, output_dim))
        if final_activation is not None:
            self.layers.append(final_activation)


    def forward_w_grad(self, x):
        x.requires_grad = True
        out = self.forward(x)
        grad = []
        for i in range(out.shape[1]):
            grad_output = torch.zeros_like(out, device=x.device)
            grad_output[:, i] = 1.0  # Isolate one output dimension at a time
            grad.append(torch.autograd.grad(outputs=out, inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
        return out, torch.stack(grad, dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RecursiveMLP(nn.Module):
    def __init__(self, input_size, output_dim, num_units, num_layers, activation, final_activation=None, stop_grad=True):
        super(RecursiveMLP, self).__init__()
        # Compute Hrev, Hfwd from input_size, output_dim
        # TODO size properly
        self.n = 3
        self.m = 3
        self.H_fwd = output_dim
        self.H_rev = (input_size - (self.n - 2) - self.H_fwd * self.m) / (1 + self.m)
        assert self.H_rev == int(self.H_rev)
        self.H_rev = int(self.H_rev)
        self.in_size = self.H_rev + (self.n - 2) + self.H_rev * self.m
        self.mlp = MLP(self.in_size, 1, num_units, num_layers, activation, final_activation)
        self.stop_grad = stop_grad

    def forward_w_grad(self, x):
        t = torch.arange(self.H_fwd, device=x.device)
        w = torch.zeros((x.shape[0], self.H_fwd), device=x.device)
        grads = []
        e = x[:, :self.H_rev]
        v = x[:, -(self.H_fwd + self.H_rev) * self.m:].reshape((x.shape[0], self.H_fwd + self.H_rev, self.m))
        for i in range(self.H_fwd):
            if i < self.H_rev:
                w_seg = torch.clone(w[:, 0:i])
                if self.stop_grad:
                    w_seg = w_seg.detach()
                data = torch.concatenate([
                    e[:, i:], w_seg,
                    v[:, i:i + self.H_rev].reshape((x.shape[0], -1)),
                    t[i] * torch.ones((x.shape[0], 1), device=x.device)
                ], dim=1)
            else:
                w_seg = torch.clone(w[:, i - self.H_rev:i])
                if self.stop_grad:
                    w_seg = w_seg.detach()
                data = torch.concatenate([
                    w_seg,
                    v[:, i:i + self.H_rev].reshape((x.shape[0], -1)),
                    t[i] * torch.ones((x.shape[0], 1), device=x.device)
                ], dim=1)
            wd, gd = self.mlp.forward_w_grad(data)
            w[:, i] = wd.squeeze(dim=1)
            grads.append(gd)
        return w, torch.cat(grads, dim=1)

def load_tube_model(artifact_name):
    api = wandb.Api()

    config, state_dict = wandb_model_load(api, artifact_name)
    H_fwd = config.dataset['H_fwd']
    H_rev = config.dataset['H_rev']

    if config.model['activation']['_target_'] == 'torch.nn.Softplus':
        act = torch.nn.Softplus(beta=config.model['activation']['beta'])
    else:
        raise RuntimeError('activation not supported')

    n = 3
    m = 3
    model = RecursiveMLP(
        input_size=H_rev  * (1 + m) + (n - 2) + H_fwd * m,
        output_dim=H_fwd,
        num_units=config.model['num_units'],
        num_layers=config.model['num_layers'],
        activation=act,
        stop_grad=config.model['stop_grad']
    )
    model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    model = load_tube_model("coleonguard-Georgia Institute of Technology/Deep_Tube_Training/gqzp1ubf_model:best")

    print('here')
