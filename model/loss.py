import torch
from ase import Atoms
import wandb

class DensityLoss:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.use_wandb = config['wandb']
        self.density_loss_fn = torch.nn.L1Loss(reduction='sum')
        self.r2loss = torch.nn.MSELoss()
        self.loss_type = config['loss_type']

    def compute_integrated_error(self, pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points=None):
        # pred_cd is either full grid (same shape as true) OR a 1D slice (len = |sampled_points|)
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)

        if sampled_points is None:
            # full-grid path
            delta = torch.abs(pred_cd - true_cd_tens).sum()
            denom = true_cd_tens.sum()
        else:
            # subsampled path: compare on the same indices, and normalize by the true sum on that slice
            t_flat = true_cd_tens.flatten()
            t_sel  = t_flat.index_select(0, sampled_points)
            p_sel  = pred_cd.flatten()
            delta  = torch.abs(p_sel - t_sel).sum()
            denom  = t_sel.sum()

        denom = denom.clamp_min(1e-12)
        NMAE = delta / denom
        return NMAE

    def compute_R2_loss(self, pred_cd: torch.Tensor, true_cd_tens: torch.Tensor, sampled_points: torch.Tensor | None = None):
        pred_cd = pred_cd.to(self.device).flatten()
        if sampled_points is None:
            true_vec = true_cd_tens.to(self.device).flatten()
        else:
            true_vec = true_cd_tens.to(self.device).flatten().index_select(0, sampled_points)
        return self.r2loss(pred_cd, true_vec)

    def compute_mae_loss(self, pred_cd: torch.Tensor, true_cd_tens: torch.Tensor, sampled_points: torch.Tensor | None = None):
        pred_cd = pred_cd.to(self.device).flatten()
        if sampled_points is None:
            true_vec = true_cd_tens.to(self.device).flatten()
        else:
            true_vec = true_cd_tens.to(self.device).flatten().index_select(0, sampled_points)
        loss = torch.abs(pred_cd - true_vec).mean()
        return loss

    def compute_total_loss(self, sys: Atoms,
                           pred_cd: torch.tensor,
                           true_cd_tens: torch.tensor,
                           grid_dict: dict,
                           n_valence_electrons: int,
                           sampled_points: torch.tensor = None,
                           training: bool = True,
                           volume: float = None):
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)
        if sampled_points is not None:
            sampled_points = sampled_points.to(self.device)

        grid_points = grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"]
        dv = (volume if volume is not None else sys.get_volume()) / grid_points

        integrated_error = self.compute_integrated_error(pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points)
        integrated_error = integrated_error.to(self.device)

        if self.loss_type == "R2":
            loss = self.compute_R2_loss(pred_cd, true_cd_tens, sampled_points)  # <- slice-aware
        elif self.loss_type == "mae":
            loss = self.compute_mae_loss(pred_cd, true_cd_tens, sampled_points)
        else:
            loss = integrated_error

        return loss, integrated_error

