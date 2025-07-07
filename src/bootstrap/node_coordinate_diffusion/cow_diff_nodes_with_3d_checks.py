import os

import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from timm import optim
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from src.ConStruct.datasets.cow_dataset import CoWGraphDataset
from src.bootstrap.node_coordinate_diffusion.sit_model import SiT_models
from src.environment_setup import PROJECT_ROOT_DIR

# Some Globals
COORDINATE_SIZE = 3


class NodeCoordsDiffusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, coordinate_size=COORDINATE_SIZE,
                 device="cuda"):
        self.model = SiT_models['SiT-T'](in_channels=COORDINATE_SIZE).to(device)
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = coordinate_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.loss_fn = nn.MSELoss()

    def to_cuda_device(self, device):
        if not next(self.model.parameters()).is_cuda:
            self.model.to(device)
            self.beta = self.beta.to(device)
            self.alpha_hat = self.alpha_hat.to(device)
            self.alpha = self.alpha.to(device)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_pos(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def compute_noise(self, pos, node_mask):
        t = self.sample_timesteps(pos.shape[0]).to(self.device)
        x_t, noise = self.noise_pos(pos, t)
        predicted_noise = self.model(x_t, t, node_mask)
        loss = self.loss_fn(noise * node_mask.unsqueeze(-1), predicted_noise * node_mask.unsqueeze(-1))
        return loss

    def sample(self, pos, node_mask):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((pos.shape[0], pos.shape[1], pos.shape[2])).to(pos.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(pos.shape[0]) * i).long().to(pos.device)
                predicted_noise = self.model(x, t, node_mask)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        self.model.train()
        return x


class dummy(object):
    def __init__(self):
        super(dummy, self).__init__()


def plot_3d_coordinates(coords_synth, coords_real, fig):
    # Compute KDE for each array
    coords_synth = coords_synth.cpu().numpy()
    coords_real = coords_real.cpu().numpy()

    # Convert the fourth axis into a 3D subplot
    ax3d = fig.add_subplot(144, projection='3d')  # Use subplot specifier for the fourth subplot

    ax3d.scatter(coords_real[:, 0], coords_real[:, 1], coords_real[:, 2], c='red', label='Real Points', alpha=0.5)
    ax3d.scatter(coords_synth[:, 0], coords_synth[:, 1], coords_synth[:, 2], c='green', label='Synthetic Points',
                 alpha=0.25)

    # Add labels and legend
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()
    ax3d.set_title("3D Point Clouds")


def perform_coord_wise_kde(coords_synth, coords_real, epoch):
    colors = ['g', 'r']
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
    labels = ['synth', 'real']
    for idx, coords in enumerate([coords_synth, coords_real]):
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        x_coord_kde = gaussian_kde(x_coords.tolist())
        y_coord_kde = gaussian_kde(y_coords.tolist())
        z_coord_kde = gaussian_kde(z_coords.tolist())
        pts = np.linspace(-1, 1, 2000)
        axes[0].plot(pts, x_coord_kde(pts), label=f"KDE_x_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[0].legend()
        axes[1].plot(pts, y_coord_kde(pts), label=f"KDE_y_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[1].legend()
        axes[2].plot(pts, z_coord_kde(pts), label=f"KDE_z_coords_{labels[idx]}", color=colors[idx], alpha=0.5 ** idx)
        axes[2].legend()
    # Adding the KDE plot on full 3d coordinates as well
    plot_3d_coordinates(coords_synth, coords_real, fig)
    plt.tight_layout()
    # Save the figure for usage
    kde_plot_dir = os.path.join(PROJECT_ROOT_DIR, 'bootstrap', 'node_coordinate_diffusion', 'kde_plots_cow')
    os.makedirs(kde_plot_dir, exist_ok=True)
    plt.savefig(f'{kde_plot_dir}/kde_{epoch}.png', dpi=80)
    plt.show()


def perform_icp(point_cloud1, point_cloud2):
    # Convert to Open3D point cloud format
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)

    # Perform ICP
    threshold = 0.02  # Distance threshold
    trans_init = np.eye(4)  # Initial transformation

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply transformation to align point clouds
    pcd2.transform(reg_p2p.transformation)

    # Compute the distance between aligned point clouds
    distances = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    return np.mean(distances).item()


def get_data(args):
    data_root = os.path.join(PROJECT_ROOT_DIR, 'data', 'cow', 'pt_files')
    train_dataset = CoWGraphDataset(dataset_name='cow', split='train', root=data_root, node_jitter=False)
    val_dataset = CoWGraphDataset(dataset_name='cow', split="val", root=data_root, node_jitter=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def train(args):
    device = args.device
    train_dataloader, val_loader = get_data(args)
    # Since we are working in 3D space
    diffusion = NodeCoordsDiffusionModel(coordinate_size=COORDINATE_SIZE, device=device, noise_steps=args.noise_steps)
    optimizer = optim.AdamW(diffusion.model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        epoch_loss = ctr = 0
        for i, data in enumerate(train_dataloader):
            data = data.to(device)
            pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
            loss = diffusion.compute_noise(pos, node_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            ctr += 1
        if epoch % 100 == 99:
            with torch.no_grad():
                sampled_positions, actual_positions = [], []
                diffusion.model.eval()
                vl = val_ctr = 0
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
                    val_loss = diffusion.compute_noise(pos, node_mask)
                    vl += val_loss.item()
                    val_ctr += 1
                    # We sample a few examples to check the KDE plots.
                    sampled_pos = diffusion.sample(pos, node_mask)
                    sampled_pos = sampled_pos[node_mask]
                    sampled_positions.append(sampled_pos.view(-1, COORDINATE_SIZE))
                    true_pos = pos[node_mask]
                    actual_positions.append(true_pos.view(-1, COORDINATE_SIZE))

                point_cloud_pred = torch.cat(sampled_positions)
                point_cloud_gt = torch.cat(actual_positions)
                perform_coord_wise_kde(point_cloud_pred, point_cloud_gt, epoch=epoch)
                # We can perform some other tests as well.
                icp_loss = perform_icp(point_cloud_pred.cpu().numpy(),
                                       point_cloud_gt.cpu().numpy())
                print(f"ICP loss {icp_loss:.4f}")
                if icp_loss < best_val_loss:
                    torch.save(diffusion.model.state_dict(),
                               os.path.join(PROJECT_ROOT_DIR, 'bootstrap', 'node_coordinate_diffusion',
                                            'cow_best_coord_model.pth'))
                    print(f"New best val model saved at {epoch=}")
                    best_val_loss = icp_loss
            # Putting model back to training
            diffusion.model.train()
        total_loss = epoch_loss / ctr
        pbar.set_description(f"epoch {epoch} finished. Loss = {total_loss:.4f}")
        torch.save(diffusion.model.state_dict(),
                   os.path.join(PROJECT_ROOT_DIR, 'bootstrap', 'node_coordinate_diffusion',
                                'cow_last_coord_model.pth'))


def test(args):
    device = args.device
    _, val_loader = get_data(args)
    diffusion = NodeCoordsDiffusionModel(coordinate_size=COORDINATE_SIZE, device=device, noise_steps=args.noise_steps)
    checkpoint_loc = os.path.join(PROJECT_ROOT_DIR, 'bootstrap', 'node_coordinate_diffusion',
                                  'cow_best_coord_model.pth')
    diffusion.model.load_state_dict(torch.load(checkpoint_loc))
    diffusion.model.to(device)
    sampled_positions, actual_positions = [], []
    diffusion.model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            pos, node_mask = to_dense_batch(x=data.pos, batch=data.batch)
            sampled_pos = diffusion.sample(pos, node_mask)
            sampled_pos = sampled_pos[node_mask]
            sampled_positions.append(sampled_pos.view(-1, COORDINATE_SIZE))
            true_pos = pos[node_mask]
            actual_positions.append(true_pos.view(-1, COORDINATE_SIZE))
        perform_coord_wise_kde(torch.cat(sampled_positions), torch.cat(actual_positions), epoch='best_val')


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 2000
    args.batch_size = 64
    args.noise_steps = 1000
    args.device = "cuda"
    args.lr = 1e-5
    args.weight_decay = 0
    train(args)
    # Let us also check the best val result
    test(args)


if __name__ == '__main__':
    launch()
