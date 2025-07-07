import torch
from src.ConStruct import utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(
            remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies
        )
        self.valency = ValencyFeature()
        self.weight = WeightFeature(
            max_weight=dataset_infos.max_weight,
            atom_weights=dataset_infos.atom_weights,
        )

    def update_input_dims(self, input_dims):
        input_dims.X += 2
        input_dims.y += 1
        return input_dims

    def __call__(self, z_t):
        charge = self.charge(z_t).unsqueeze(-1)  # (bs, n, 1)
        valency = self.valency(z_t).unsqueeze(-1)  # (bs, n, 1)
        placeholder_X = torch.cat((charge, valency), dim=-1)
        placeholder_E = torch.zeros((*z_t.E.shape[:-1], 0)).type_as(z_t.E)
        placeholder_y = self.weight(z_t)  # (bs, 1)

        return utils.PlaceHolder(
            X=placeholder_X,
            E=placeholder_E,
            y=placeholder_y,
        )


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, z_t):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(
            1, 1, 1, -1
        )
        weighted_E = z_t.E * bond_orders  # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)  # (bs, n)

        valencies = torch.tensor(self.valencies, device=z_t.X.device).reshape(1, 1, -1)
        X = z_t.X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)  # (bs, n)

        return (normal_valencies - current_valencies).to(z_t.X.device)


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, z_t):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=z_t.E.device).reshape(
            1, 1, 1, -1
        )
        E = z_t.E * orders  # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)  # (bs, n)
        return valencies.to(z_t.X.device)


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.Tensor(atom_weights)

    def __call__(self, z_t):
        X = torch.argmax(z_t.X, dim=-1)  # (bs, n)
        X_weights = self.atom_weight_list.to(X.device)[X]  # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1) / self.max_weight  # (bs, 1)
