import math
import random
from functools import partial
from typing import Optional, Tuple, Union
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import functional as F

import dist
from models.basic_var import AdaLNSABlock, SABlock
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class ControlVAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, norm_eps=1e-6, aln=1, aln_gamma_init=1e-3, shared_aln=False, cond_drop_rate=0.1,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        layer_scale=-1., tau=4, cos_attn=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True, mask_factor=2, bidirectional=False, separate_decoding=False,
        separator=False, type_pos=False, indep=True, multi_cond=False,
    ):
        super().__init__()
        # 0. hyperparameters
        cos_attn = True if depth == 30 else False
        if cos_attn:
            print(f'Rewrite cos_attn to True when depth={depth}')
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.using_aln, self.aln_init, self.aln_gamma_init, self.layer_scale = aln >= 0, aln, aln_gamma_init, layer_scale
        if self.using_aln and layer_scale != -1:
            print(f'**WARNING**: using AdaLNSABlock with {aln=:g}, {aln_gamma_init=:g}; the arg {layer_scale=:g} will be IGNORED because only SABlock cares about layer_scale', flush=True)

        self.separator = separator
        self.bidirectional = bidirectional
        self.separate_decoding = separate_decoding
        self.type_pos = type_pos
        self.indep = indep
        self.multi_cond = multi_cond

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.mask_factor = mask_factor
        self.L = sum(pn ** 2 * mask_factor for pn in self.patch_nums)  # image mask pair
        if self.separator:
            self.L += (len(self.patch_nums) - 1) * mask_factor  # special tokens
        self.first_l = self.patch_nums[0] ** 2 * mask_factor
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if i != 0 and self.separator else 0
            self.begin_ends.append((cur, cur+(pn ** 2 + num_sp_tokens) * mask_factor))
            cur += (pn ** 2 + num_sp_tokens) * mask_factor
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if i != 0 and self.separator else 0
            pe = torch.empty(1, (pn*pn + num_sp_tokens) * mask_factor, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        # TODO: test separate mask/image level embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)  #  lvl_1L = mT[:, 0].contiguous()
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        if self.type_pos:
            self.type_embed = nn.Embedding(self.mask_factor, self.C)  # lvl_1L = mT[:, 0].contiguous()
            nn.init.trunc_normal_(self.type_embed.weight.data, mean=0, std=init_std)
            print('Creating type positional encoding')
            m, m_ = [], []
            for i, pn in enumerate(self.patch_nums):
                num_sp_tokens = 1 if (i != 0 and self.separator) else 0
                m.append(torch.full((pn*pn + num_sp_tokens,), 1))
                m.append(torch.full((pn * pn + num_sp_tokens,), 0))
                m_.append(torch.full((pn * pn + num_sp_tokens,), 0))
                m_.append(torch.full((pn * pn + num_sp_tokens,), 1))
            m = torch.cat(m).view(1, self.L, 1)
            m_ = torch.cat(m_).view(1, self.L, 1)
            mT = m.transpose(1, 2)  # dT: 11L
            mT_ = m_.transpose(1, 2)  # dT: 11L
            type_1L = mT[:, 0].contiguous()
            type_1L_ = mT_[:, 0].contiguous()
            self.register_buffer('type_1L', type_1L)
            self.register_buffer('type_1L_', type_1L_)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln and self.using_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSABlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            ) if self.using_aln else SABlock(
                layer_scale=layer_scale,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        if self.blocks[-1].fused_add_norm_fn is not None:
            self.gamma2_last = nn.Parameter(self.layer_scale * torch.ones(embed_dim), requires_grad=True) if self.layer_scale >= 0 else 1
        else:
            self.gamma2_last = None

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [vGPT config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d = []
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if (i != 0 and self.separator) else 0
            d.append(torch.full(((pn*pn + num_sp_tokens) * mask_factor,), i))
        d: torch.Tensor = torch.cat(d).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L

        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

        if separate_decoding:
            # ignore the upper right half in each stage
            d = []
            dT = []
            for i, pn in enumerate(self.patch_nums):
                num_sp_tokens = 1 if i != 0 and self.separator else 0
                d.extend([torch.full((pn*pn + num_sp_tokens,), 1 + 4 * i,), torch.full((pn*pn + num_sp_tokens,), 3 + 4 * i,)])
                dT.extend([torch.full((pn*pn + num_sp_tokens,), 1 + 4 * i, ), torch.full((pn*pn + num_sp_tokens,), 2 + 4 * i, )])
            d = torch.cat(d).view(1, self.L, 1)
            dT = torch.cat(dT).view(1, 1, self.L)
            attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

            if self.indep:
                d = []
                dT = []
                for i, pn in enumerate(self.patch_nums):
                    num_sp_tokens = 1 if i != 0 and self.separator else 0
                    d.extend([torch.full((pn * pn + num_sp_tokens,), 3 + 4 * i, ), torch.full((pn * pn + num_sp_tokens,), 1 + 4 * i, )])
                    dT.extend([torch.full((pn * pn + num_sp_tokens,), 2 + 4 * i, ), torch.full((pn * pn + num_sp_tokens,), 0 + 4 * i, )])
                d = torch.cat(d).view(1, self.L, 1)
                dT = torch.cat(dT).view(1, 1, self.L)
                attn_bias_for_masking += torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

                # attn_bias_for_masking_ = torch.where(attn_bias_for_masking == 0, 0., 255).reshape(1, 1, self.L, self.L)
                # import numpy as np
                # from PIL import Image
                # Image.fromarray(attn_bias_for_masking_.cpu().numpy().astype(np.uint8)[0, 0]).convert('L').save('mask_.png')

        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        num_total_sp_tokens = self.num_stages_minus_1 * mask_factor if self.separator else 0
        if self.using_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
            self.head = nn.Linear(self.C, self.V + num_total_sp_tokens)
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.V + num_total_sp_tokens))
        if self.separator:
            self.special_embed = nn.Embedding(self.num_stages_minus_1 * self.mask_factor, self.C)  # skip the first stage
            nn.init.trunc_normal_(self.special_embed.weight.data, mean=0, std=init_std)
        if self.multi_cond:
            self.cond_embed = nn.Embedding(5, self.C)
            nn.init.trunc_normal_(self.cond_embed.weight.data, mean=0, std=init_std)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # is h_and_residual, so fused_add_norm must be used, so self.gamma2_last is not None
            h = resi + self.gamma2_last * self.blocks[-1].drop_path(h)
        else:   # is h, so fused_add_norm is not used, and self.gamma2_last is None
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def conditional_infer_cfg(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None, cfg=(1.5, 1.5, 1.5), top_k=0, top_p=0.0,
            more_smooth=False, cond_type=None, c_mask=None, c_img=None,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        if label_B is None:
            label_B = torch.multinomial(self.selecting_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)
        empty_cls = torch.full_like(label_B, fill_value=self.num_classes)
        # p(c2|c1,C,I)p(c1|C,I)p(C|I)p(I)
        # label_B = torch.cat((label_B, empty_cls, empty_cls), dim=0)
        # p(c1|c2,C,I)p(c2|C,I)p(C|I)p(I)
        label_B = torch.cat((label_B, empty_cls, empty_cls, empty_cls), dim=0)
        sos = cond_BD = self.class_emb(label_B)

        empty_cond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device).long()
        # p(c2|c1,C,I)p(c1|C,I)p(C|I)p(I)
        # cond_type = torch.concat([empty_cond_type, empty_cond_type, empty_cond_type], dim=0)
        # p(c1|c2,C,I)p(c2|C,I)p(C|I)p(I)
        cond_type = torch.concat([cond_type, cond_type, empty_cond_type, empty_cond_type], dim=0)
        sos = sos.unsqueeze(1)
        cond_token = self.cond_embed(cond_type).unsqueeze(1)
        next_token_map = torch.concat([cond_token, sos], dim=1)

        repeat_num = label_B.shape[0] // B
        next_token_map = next_token_map + self.pos_start.expand(repeat_num * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        for b in self.blocks: b.attn.kv_caching(True)

        cur_L = 0
        num_sp_token = 0
        f_hat = sos.new_zeros(repeat_num * B, self.Cvae, self.patch_nums[-1] * self.mask_factor, self.patch_nums[-1])
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += (pn * pn + num_sp_token) * self.mask_factor
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            SABlock.forward
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None if not self.indep else
                self.attn_bias_for_masking[:, :, (cur_L - (pn * pn + num_sp_token) * self.mask_factor):cur_L, :cur_L])
            logits_BlV = self.get_logits(x, cond_BD)
            # class, cond_type, pixel_cond
            # [c1, c2, C], [x, c2, C], [x, x, C], [x, x, x]
            t1, t2, t3 = cfg[0] * ratio, cfg[1] * ratio, cfg[2] * ratio

            if repeat_num == 4:
                # logits_BlV = t1 * logits_BlV[:B] \
                #              + (t2 - t1) * logits_BlV[B:2 * B] \
                #              + (t3 - t2) * logits_BlV[2 * B:3 * B] \
                #              + (1 - t3) * logits_BlV[-B:]
                logits_BlV = (1 + t1) * logits_BlV[:B] \
                             + (t2 - t1) * logits_BlV[B:2 * B] \
                             + (t3 - t2) * logits_BlV[2 * B:3 * B] \
                             - t3 * logits_BlV[-B:]
            elif repeat_num == 3:
                # logits_BlV = t1 * logits_BlV[:B] \
                #              + (t2 - t1) * logits_BlV[B:2 * B] \
                #              + (1 - t2) * logits_BlV[-B:]
                logits_BlV = (1 + t1) * logits_BlV[:B] \
                             + (t2 - t1) * logits_BlV[B:2 * B] \
                             - t2 * logits_BlV[-B:]
            logits_BlV = logits_BlV.repeat(repeat_num, 1, 1)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if c_mask is not None :  # Teaching force
                if repeat_num == 4:
                    idx_Bl[:B, :pn * pn] = c_mask[si]
                    idx_Bl[B:2 * B, :pn * pn] = c_mask[si]
                    idx_Bl[2 * B:3 * B, :pn * pn] = c_mask[si]
                elif repeat_num == 3:
                    idx_Bl[:B, :pn * pn] = c_mask[si]
                    idx_Bl[B:2 * B, :pn * pn] = c_mask[si]
            if c_img is not None:
                if repeat_num == 4:
                    idx_Bl[:B, pn * pn:] = c_img[si]
                    idx_Bl[B:2 * B, pn * pn:] = c_img[si]
                    idx_Bl[2 * B:3 * B, pn * pn:] = c_img[si]
                elif repeat_num == 3:
                    idx_Bl[:B, pn * pn:] = c_img[si]
                    idx_Bl[B:2 * B, pn * pn:] = c_img[si]

            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1,
                                                 rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            assert self.mask_factor == 2, 'current visualization only support mask_factor == 2'
            h_BChw = h_BChw.transpose_(1, 2)
            h_BChw_1 = h_BChw[:, :, :pn * pn].reshape(repeat_num * B, self.Cvae, pn, pn)  # first part
            h_BChw_2 = h_BChw[:, :, -pn * pn:].reshape(repeat_num * B, self.Cvae, pn, pn)  # second part
            f_hat_1 = f_hat[:, :, :self.patch_nums[-1], :]
            f_hat_2 = f_hat[:, :, self.patch_nums[-1]:, :]
            f_hat_1, next_token_map_1 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_1, h_BChw_1)
            f_hat_2, next_token_map_2 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_2, h_BChw_2)
            f_hat = torch.concat((f_hat_1, f_hat_2), dim=2)  # [b, c, 2pn, pn]
            next_token_map_1 = next_token_map_1.view(repeat_num * B, self.Cvae, -1).transpose(1, 2)
            next_token_map_2 = next_token_map_2.view(repeat_num * B, self.Cvae, -1).transpose(1, 2)
            next_token_map = torch.concat((next_token_map_1, next_token_map_2), dim=1)  # [b, c, 2pn, pn]
            next_token_map = self.word_embed(next_token_map)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2) * self.mask_factor]

        f_hat_1 = f_hat_1[:B]
        f_hat_2 = f_hat_2[:B]
        for b in self.blocks: b.attn.kv_caching(False)
        img1 = self.vae_proxy[0].fhat_to_img(f_hat_1).add_(1).mul_(0.5)
        img2 = self.vae_proxy[0].fhat_to_img(f_hat_2).add_(1).mul_(0.5)
        return torch.concat([img1, img2], dim=2)  # de-normalize, from [-1, 1] to [0, 1]

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, cond_type=None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.selecting_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        mask_first = True
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        if self.multi_cond:
            if cond_type is None:
                if B == 4:
                    cond_type = torch.tensor([0, 1, 2, 3], device=dist.get_device())
                    uncond_type = torch.tensor([4, 4, 4, 4], device=dist.get_device())
                else:
                    cond_idx = torch.full((1, 4), fill_value=1 / 4, dtype=torch.float32, device=dist.get_device())
                    cond_type = torch.multinomial(cond_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
                    uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device)
            elif isinstance(cond_type, int):
                assert cond_type <= 3 and cond_type > 0
                cond_type = torch.full((B,), fill_value=cond_type, device=self.lvl_1L.device)
                uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device)
            else:
                uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device).long()
            cond_type = torch.concat([cond_type, uncond_type], dim=0)  # copy for cfg
            sos = sos.unsqueeze(1)
            cond_token = self.cond_embed(cond_type).unsqueeze(1)
            mask_first = True if (random.random() < 0.5 or not self.bidirectional) else False
            if mask_first:
                next_token_map = torch.concat([cond_token, sos], dim=1)
            else:
                next_token_map = torch.concat([sos, cond_token], dim=1)

            next_token_map = next_token_map + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        else:
            if self.bidirectional and self.mask_factor == 2:  # random shuffle mask and image sos
                ch_sign = sos.new_ones(2 * B, self.first_l // 2, 1)
                sign = random.choice([-1, 1])
                ch_sign = torch.cat([ch_sign * sign, -ch_sign * sign], dim=1)
                mask_first = True if sign == 1 else False
                next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) * ch_sign + \
                                 self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
            else:
                next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + \
                                 self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        if self.type_pos:
            type_pos = self.type_embed(self.type_1L.expand(B, -1)) if mask_first else self.type_embed(self.type_1L_.expand(B, -1))

        for b in self.blocks: b.attn.kv_caching(True)

        if self.separate_decoding and not self.indep:
            cur_L = 0
            next_token_map_1 = next_token_map[:, :self.patch_nums[0]]
            next_token_map_2 = next_token_map[:, self.patch_nums[0]:]
            f_hat_1 = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
            f_hat_2 = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
            iter_patch_nums = list(chain.from_iterable(zip(self.patch_nums, self.patch_nums)))
            num_sp_token = 0
            for si, pn in enumerate(iter_patch_nums):  # si: i-th segment
                ratio = (si // 2) / self.num_stages_minus_1
                cur_L += pn * pn + num_sp_token
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                SABlock.forward
                if si == 0:
                    x = next_token_map_1
                elif si == 1:
                    x = next_token_map_2
                else:
                    x = next_token_map
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                logits_BlV = logits_BlV[:, :, :self.V]  # ignore special tokens
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

                if si > 1 and self.separator:
                    idx_Bl = idx_Bl[:, :-1]  # discard special token if used
                    num_sp_token = 1
                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1,
                                                     rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                if si % 2 == 0:
                    f_hat_1, _ = self.vae_quant_proxy[0].get_next_autoregressive_input(si//2, len(self.patch_nums), f_hat_1, h_BChw)
                    next_token_map = F.interpolate(f_hat_1, size=(iter_patch_nums[si+1], iter_patch_nums[si+1]), mode='area')
                else:
                    f_hat_2, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si//2, len(self.patch_nums), f_hat_2, h_BChw)

                if si != len(iter_patch_nums) - 1:  # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    if self.separator and si >= 1:
                        # teaching force
                        mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                        special_token = self.special_embed(torch.full((B,), fill_value=mapping[si-1], device=sos.device, dtype=torch.long))
                        next_token_map = torch.concat((self.word_embed(next_token_map), special_token.unsqueeze(1)), dim=1)
                    else:
                        next_token_map = self.word_embed(next_token_map)
                    next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + iter_patch_nums[si + 1] ** 2 + num_sp_token]
                    if self.type_pos:
                        next_token_map = next_token_map + type_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2 + num_sp_token) * self.mask_factor]
                    next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

        else:
            cur_L = 0
            num_sp_token = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1] * self.mask_factor, self.patch_nums[-1])
            for si, pn in enumerate(self.patch_nums):   # si: i-th segment
                ratio = si / self.num_stages_minus_1
                cur_L += (pn*pn + num_sp_token) * self.mask_factor
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                SABlock.forward
                x = next_token_map
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None if not self.indep else
                    self.attn_bias_for_masking[:, :, (cur_L-(pn * pn + num_sp_token) * self.mask_factor):cur_L, :cur_L])
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                logits_BlV = logits_BlV[:, :, :self.V]  # ignore special tokens if used
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

                if si > 1 and self.separator:
                    idx_Bl_ = torch.concat((idx_Bl[:, :pn*pn], idx_Bl[:, pn*pn + 1:pn*pn*2 + 1],), dim=1)  # remove special tokens
                    idx_Bl = idx_Bl_

                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

                assert self.mask_factor <= 2, 'current visualization only support mask_factor == 2 or 1'
                if self.mask_factor == 1:
                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
                elif self.mask_factor == 2:
                    h_BChw = h_BChw.transpose_(1, 2)
                    h_BChw_1 = h_BChw[:, :, :pn*pn].reshape(B, self.Cvae, pn, pn)  # first part
                    h_BChw_2 = h_BChw[:, :, -pn*pn:].reshape(B, self.Cvae, pn, pn)  # second part
                    f_hat_1 = f_hat[:, :, :self.patch_nums[-1], :]
                    f_hat_2 = f_hat[:, :, self.patch_nums[-1]:, :]
                    f_hat_1, next_token_map_1 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_1, h_BChw_1)
                    f_hat_2, next_token_map_2 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_2, h_BChw_2)
                    f_hat = torch.concat((f_hat_1, f_hat_2), dim=2)  # [b, c, 2pn, pn]
                    next_token_map = torch.concat((next_token_map_1, next_token_map_2), dim=2)
                else:
                    raise NotImplementedError

                if si != self.num_stages_minus_1:   # prepare for next stage
                    if self.mask_factor == 1:
                        next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)

                    elif self.mask_factor == 2:
                        next_token_map_1, next_token_map_2 = next_token_map[:, :, :pn, :], next_token_map[:, :, pn:, :]
                        next_token_map_1 = next_token_map_1.view(B, self.Cvae, -1).transpose(1, 2)
                        next_token_map_2 = next_token_map_2.view(B, self.Cvae, -1).transpose(1, 2)

                        if self.separator:
                            mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                            label1, label2 = mapping[2 * si] + self.V, mapping[2 * si + 1] + self.V
                            label1, label2 = sos.new_ones(B, ) * label1, sos.new_ones(B, ) * label2
                            label1, label2 = label1.unsqueeze(1), label2.unsqueeze(1)
                            special_token1, special_token2 = self.special_embed(label1.long()), self.special_embed(label2.long())
                            next_token_map_1 = self.word_embed(next_token_map_1)
                            next_token_map_2 = self.word_embed(next_token_map_2)
                            next_token_map = torch.concat((next_token_map_1, special_token1, next_token_map_2, special_token2), dim=1)
                            num_sp_token = 1
                        else:
                            next_token_map = torch.concat((next_token_map_1, next_token_map_2), dim=1)  # [b, c, 2pn, pn]
                            next_token_map = self.word_embed(next_token_map)

                    next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + (self.patch_nums[si+1] ** 2 + num_sp_token) * self.mask_factor]
                    if self.type_pos:
                        next_token_map = next_token_map + type_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2 + num_sp_token) * self.mask_factor]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.attn.kv_caching(False)
        img1 = self.vae_proxy[0].fhat_to_img(f_hat_1).add_(1).mul_(0.5)
        img2 = self.vae_proxy[0].fhat_to_img(f_hat_2).add_(1).mul_(0.5)
        return torch.concat([img1, img2], dim=2)   # de-normalize, from [-1, 1] to [0, 1]


    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, cond_type, mask_first=True) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        with torch.autocast(device_type=label_B.device.type, enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)

            if self.multi_cond and self.mask_factor == 2:
                sos = sos.unsqueeze(1).expand(B, 1, -1)
                # 0: mask, 1: canny, 2: depth, 3: normal, 4: uncond
                cond_type = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, 4, cond_type)
                cond_token = self.cond_embed(cond_type)
                cond_token = cond_token.unsqueeze(1).expand(B, 1, -1)
                sos = torch.concat([cond_token, sos], dim=1) if mask_first else torch.concat([sos, cond_token], dim=1)
                sos = sos + self.pos_start.expand(B, self.first_l, -1)

            else:
                if self.bidirectional and self.mask_factor == 2:  # random shuffle mask and image sos
                        sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
                        ch_sign = sos.new_ones(B, self.first_l // 2, 1)
                        sign = -1 if mask_first else 1
                        ch_sign = torch.cat([ch_sign * sign, -ch_sign * sign], dim=1)
                        sos = sos * ch_sign
                else:
                    sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                if self.separator:
                    mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                    x_BLC = self.word_embed(x_BLCv_wo_first_l.float())
                    new_x = [sos,]
                    cur = 0
                    for si, pn in enumerate(self.patch_nums[1:]):  # skip first
                        label1, label2 = mapping[2 * si] + self.V, mapping[2 * si+1] + self.V
                        label1, label2 = x_BLC.new_ones(B,) * label1, x_BLC.new_ones(B,) * label2
                        label1, label2 = label1.unsqueeze(1), label2.unsqueeze(1)
                        special_token1, special_token2 = self.special_embed(label1.long()), self.special_embed(label2.long())
                        x1 = x_BLC[:, cur: cur + pn * pn]
                        x2 = x_BLC[:, cur + pn*pn: cur + pn*pn * self.mask_factor]
                        new_x.extend([x1, special_token1, x2, special_token2])
                        cur += pn*pn * self.mask_factor
                    assert cur == x_BLCv_wo_first_l.shape[1]
                    x_BLC = torch.cat(new_x, dim=1)
                else:
                    x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
            if self.type_pos:
                x_BLC += self.type_embed(self.type_1L[:, :ed].expand(B, -1)) if mask_first else self.type_embed(self.type_1L_[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        SABlock.forward, AdaLNSABlock.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size

    def special_init(self, hd0: float): # hd0: head init scale
        if hd0 >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(hd0)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(hd0)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            if True:
                self.head_nm.ada_lin[-1].weight.data.mul_(self.aln_init)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: Union[AdaLNSABlock, SABlock]
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(self.aln_gamma_init)
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(self.aln_init)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2].mul_(self.aln_gamma_init)
                sab.ada_gss.data[:, :, 2:].mul_(self.aln_init)

    def extra_repr(self):
        gamma2_last = self.gamma2_last
        if isinstance(gamma2_last, nn.Parameter):
            gamma2_last = f'<vector {self.layer_scale}>'
        return f'drop_path_rate={self.drop_path_rate:g}, layer_scale={self.layer_scale:g}, gamma2_last={gamma2_last}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x
