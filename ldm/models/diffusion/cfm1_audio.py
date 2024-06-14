import os
from pytorch_memlab import LineProfiler,profile
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from torchvision.utils import make_grid
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities import rank_zero_only # torch2
from torchdyn.core import NeuralODE
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.models.diffusion.ddpm_audio import LatentDiffusion_audio, disabled_train
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from omegaconf import ListConfig

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4

    def p_losses(self, x_start, cond, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        ut = x1 - (1 - self.sigma_min) * x0  # 和ut的梯度没关系
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.sigma_min) * t_unsqueeze) * x0

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        target = ut

        mean_dims = list(range(1,len(target.shape)))
        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss = loss_simple
        loss = self.l_simple_weight * loss.mean()
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper(cond), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self, cond)

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.channels > 0:
                shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            else:
                shape = (batch_size, self.mel_dim, self.mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning)


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        # if use_original_steps:
        #     sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        #     sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        # else:
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


class Wrapper(nn.Module):
    def __init__(self, net, cond):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        return self.net.apply_model(x, t, self.cond)


class Wrapper_cfg(nn.Module):

    def __init__(self, net, cond, unconditional_guidance_scale, unconditional_conditioning):
        super(Wrapper_cfg, self).__init__()
        self.net = net
        self.cond = cond
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def forward(self, t, x, args):
        x_in = torch.cat([x] * 2)
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        t_in = torch.cat([t] * 2)
        c_in = torch.cat([self.unconditional_conditioning, self.cond])  # c/uc shape [b,seq_len=77,dim=1024],c_in shape [b*2,seq_len,dim]
        e_t_uncond, e_t = self.net.apply_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        return e_t


class CFM_inpaint(CFM):

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        x = batch[k]
        if self.channels > 0:  # use 4d input
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()

        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', 'hybrid_feat']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            ##### Testing #######
            spec = xc['mix_spec'].to(self.device)
            encoder_posterior = self.encode_first_stage(spec)
            z_spec = self.get_first_stage_encoding(encoder_posterior).detach()
            c = {"mix_spec": z_spec, "mix_video_feat": xc['mix_video_feat']}
            ##### Testing #######
            if bs is not None:
                c = {"mix_spec": c["mix_spec"][:bs], "mix_video_feat": c['mix_video_feat'][:bs]}
            # Testing #
            if cond_key == 'masked_image':
                mask = super().get_input(batch, "mask")
                cc = torch.nn.functional.interpolate(mask, size=c.shape[-2:]) # [B, 1, 10, 106]
                c = torch.cat((c, cc), dim=1) # [B, 5, 10, 106]
            # Testing #
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out


    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn" or self.model.conditioning_key == "hybrid_inpaint":
                key = "c_crossattn"
            else:
                key = "c_film"
            cond = {key: cond}


        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon



    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N) # z is latent,c is condition embedding, xc is condition(caption) list
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x if len(x.shape)==4 else x.unsqueeze(1)
        log["reconstruction"] = xrec if len(xrec.shape)==4 else xrec.unsqueeze(1)
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode") and self.cond_stage_key != "masked_image":
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key == "masked_image":
                log["mask"] = c[:, -1, :, :][:, None, :, :]
                xc = self.cond_stage_model.decode(c[:, :self.cond_stage_model.embed_dim, :, :])
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                pass
                # xc = log_txt_as_img((256, 256), batch["caption"])
                # log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            if len(diffusion_row[0].shape) == 3:
                diffusion_row = [x.unsqueeze(1) for x in diffusion_row]
            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
