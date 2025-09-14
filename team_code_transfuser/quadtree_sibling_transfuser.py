# quadtree_mxcif_mixer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

# -----------------------------
# utilities
# -----------------------------
def _norm_hw(patch_hw: Union[int, Tuple[int,int]], Lpatch: int) -> Tuple[int,int]:
    if isinstance(patch_hw, tuple):
        H, W = int(patch_hw[0]), int(patch_hw[1])
        if H * W != Lpatch:
            raise ValueError(f"patch_hw=({H},{W}) but Lpatch={Lpatch}.")
        return H, W
    elif isinstance(patch_hw, int):
        H = W = int(patch_hw)
        if H * W != Lpatch:
            raise ValueError(f"patch_hw={H} but Lpatch={Lpatch}.")
        return H, W
    else:
        root = int(round(math.sqrt(Lpatch)))
        if root * root != Lpatch:
            raise ValueError(f"Cannot infer square grid from Lpatch={Lpatch}.")
        return root, root

def build_sibling_buffers(H: int, W: int):
    """Return:
       idxbuf: [L, Nmax, 4] long indices into base H*W grid
       mskbuf: [L, Nmax] bool mask (valid groups per level)
    """
    levels = []
    h, w = H, W
    while (h > 1) or (w > 1):
        nH, nW = (h + 1) // 2, (w + 1) // 2
        groups = []
        for i in range(nH):
            for j in range(nW):
                r0 = min(2 * i + 0, h - 1)
                r1 = min(2 * i + 1, h - 1)
                c0 = min(2 * j + 0, w - 1)
                c1 = min(2 * j + 1, w - 1)
                base = lambda r, c: r * w + c
                groups.append([base(r0, c0), base(r0, c1), base(r1, c0), base(r1, c1)])
        levels.append(torch.tensor(groups, dtype=torch.long))
        h, w = nH, nW

    if len(levels) == 0:
        # already 1x1; make a dummy empty level
        return torch.zeros(0, 0, 4, dtype=torch.long), torch.zeros(0, 0, dtype=torch.bool)

    L = len(levels)
    Nmax = max(buf.size(0) for buf in levels)
    idxbuf = torch.full((L, Nmax, 4), -1, dtype=torch.long)
    mskbuf = torch.zeros((L, Nmax), dtype=torch.bool)
    for i, buf in enumerate(levels):
        idxbuf[i, :buf.size(0)] = buf
        mskbuf[i, :buf.size(0)] = 1
    return idxbuf, mskbuf

# -----------------------------
# MX‑CiF pairwise (fixed‑k) on 2x2 siblings
# -----------------------------
class MxCIFSiblingAttentionPairwise(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_keep: int = 3, drop_ffn: float = 0.10):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh, self.dh = n_heads, d_model // n_heads
        self.k_keep = int(max(1, min(4, k_keep)))

        self.pre_ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.cif_proj = nn.Linear(d_model, d_model)
        self.out_scale = nn.Parameter(torch.tensor([0.35], dtype=torch.float32))
        self.local_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(drop_ffn),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(drop_ffn),
        )

        # for optional parent debug/outside use
        self.proj_parent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model, bias=False))

        self.register_buffer("_temp", torch.tensor(1.0), persistent=False)

    @torch.no_grad()
    def set_temp(self, t: float):
        self._temp.fill_(float(max(1e-3, t)))

    def _mask_topk_fixed(self, scores):  # [B,N,H,4,4]
        if self.k_keep >= 4:
            out = scores
        else:
            topv, topi = torch.topk(scores, k=self.k_keep, dim=-1)
            with torch.no_grad():
                mu  = scores.detach().mean(dim=-1, keepdim=True)
                std = scores.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
                floor = mu - 6.0 * std
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, topi, True)
            out = torch.where(mask, scores, floor)
        out = out / float(self._temp.item())
        return torch.clamp(out, min=-50.0, max=50.0)

    def forward(self, x_grp: torch.Tensor, key_bias: Optional[torch.Tensor] = None):
        """
        x_grp: [B, n, 4, D]; returns (o_parent: [B,n,D], x_grp_out: [B,n,4,D])
        """
        B, N, C, D = x_grp.shape
        H = self.nh; dh = self.dh
        x_in = self.pre_ln(x_grp)
        qkv = self.qkv(x_in.reshape(-1, D)).view(B, N, C, 3, H, dh)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        qh, kh, vh = (t.permute(0, 1, 3, 2, 4).contiguous() for t in (q, k, v))  # [B,N,H,4,dh]
        scale = dh ** -0.5

        ###################### torch.amp -> torch.cuda.amp ######################
        with torch.cuda.amp.autocast(enabled=False):
            qh32, kh32, vh32 = qh.float(), kh.float(), vh.float()
            scores = torch.einsum('bnhcd,bnhkd->bnhck', qh32, kh32) * scale  # [B,N,H,4,4]
            if key_bias is not None:
                if key_bias.dim() == 3:   # [B,N,4]
                    kb = key_bias.float().unsqueeze(2).unsqueeze(3)  # [B,N,1,1,4]
                else:                      # [B,N,H,4]
                    kb = key_bias.float().unsqueeze(3)               # [B,N,H,1,4]
                scores = scores + kb

            scores = self._mask_topk_fixed(scores)
            scores = scores - scores.amax(dim=-1, keepdim=True)
            w = torch.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            mix = torch.einsum('bnhck,bnhkd->bnhcd', w, vh32)
            child_upd = mix.permute(0, 1, 3, 2, 4).contiguous().view(B, N, 4, D).to(x_grp.dtype)

        gate = torch.sigmoid(self.cif_proj(x_grp.reshape(-1, D))).view(B, N, 4, D)
        x_grp_out = x_grp + torch.clamp(self.out_scale, 0.05, 1.20) * gate * (child_upd - x_grp)
        x_ffn = self.local_ffn(x_grp_out.reshape(-1, D)).view(B, N, 4, D)
        x_grp_out = x_grp_out + 0.5 * x_ffn

        o_parent = self.proj_parent(x_grp_out.mean(dim=2))  # [B,n,D]
        return o_parent, x_grp_out

# -----------------------------
# QuadtreeMixerCLS: Generate root and replace CLS
# -----------------------------
class QuadtreeMixerCLS(nn.Module):
    """
    Complete MX‑CiF‑Quadtree (fixed‑k3) module, lightweight integration by "replacing CLS":
      forward(x) -> x_out  (same shape as x [B, L, D], but the 0th token is replaced by root)
    Parameters:
      - d_model, n_heads
      - patch_hw: int or (H, W); if None, automatically inferred as square from Lpatch
      - levels: None means use all quadtree levels; or specify an integer to use the first several levels
      - k_keep: default is 3 (fixed‑k3)
      - tau_min: temperature lower bound (recommend 1.10)
      - prefix_tokens: number of prefix tokens at the beginning of the sequence (usually 1=CLS; TransFuser can set to 0 if no CLS)
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 patch_hw: Union[int, Tuple[int,int], None],
                 levels: Optional[int] = None,
                 k_keep: int = 3,
                 tau_min: float = 1.10,
                 prefix_tokens: int = 1,
                 use_group_norm: bool = True):
        
        
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.user_patch_hw = patch_hw
        self.prefix_tokens = int(prefix_tokens)
        self.tau_min = float(tau_min)
        self.use_group_norm = bool(use_group_norm)
        self.group_norm = nn.LayerNorm(d_model)

         # buffers are constructed in the first forward pass according to (H,W
        self.register_buffer('sib_idx', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('sib_msk', torch.empty(0, dtype=torch.bool), persistent=False)
        self._built_for = None  # (H,W)
        self.levels_wanted = levels  # None -> use all

        # per-level pairwise
        self.level_attn = nn.ModuleList()

        # parent-bank weighted pooling
        self.parent_gate = nn.Linear(d_model, 1)
        self.out_norm = nn.LayerNorm(d_model)

        # attention temperature and pooling temperature
        self._tau_attn = 1.5
        self._tau_pool = 1.5
        self.k_keep = int(k_keep)

    @torch.no_grad()
    def set_temp_all(self, t: float):
        """set soft temperature in pairwise"""
        self._tau_attn = float(t)
        for m in self.level_attn:
            m.set_temp(t)

    @torch.no_grad()
    def set_pool_temp(self, t: float):
        """set soft temperature in pooling"""
        self._tau_pool = float(t)

    def _ensure_buffers(self, device, H, W):
        if self._built_for == (H, W):
            return
        idx, msk = build_sibling_buffers(H, W)
        L = idx.size(0)
        self.sib_idx = idx.to(device)
        self.sib_msk = msk.to(device)

        L_use = L if self.levels_wanted is None else int(min(self.levels_wanted, L))
        self.level_attn = nn.ModuleList([
            MxCIFSiblingAttentionPairwise(self.d_model, self.n_heads, k_keep=3)  # fixed‑k3
            for _ in range(L_use)
        ])

        ###################### MOVE TO DEVICE ######################
        self.level_attn = self.level_attn.to(device)
        for i, module in enumerate(self.level_attn):
            module = module.to(device)
            for submodule in module.modules():
                submodule.to(device)
        ############################################################

        # synchronize current temperature to newly created modules
        for m in self.level_attn:
            m.set_temp(self._tau_attn)

        self._built_for = (H, W)

    def _weighted_pool(self, parent_bank: list):
        """parent_bank: list of [B, n_l, D]"""
        if len(parent_bank) == 0:
            # No layers to aggregate (e.g. 1x1 grid), directly return zero vector (do not change CLS)
            return None
        feats = torch.cat(parent_bank, dim=1)              # [B, T, D]
        tau = max(self.tau_min, float(self._tau_pool))
        gate = self.parent_gate(feats).squeeze(-1)         # [B, T]
        w = torch.softmax(gate / tau, dim=1).unsqueeze(-1) # [B, T, 1]
        root = (w * feats).sum(1) + 0.1 * feats.mean(1)    # [B, D]
        return self.out_norm(root)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D], where the first prefix_tokens are prefix (usually CLS).
        Returns: x_out with the same shape as x, but the 0th token is replaced by root (if prefix_tokens==0, root is inserted at the first position and the last patch is trimmed to keep the length unchanged).
        """
        B, L, D = x.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expect {self.d_model}"
        assert L > self.prefix_tokens, "Sequence must contain patch tokens after prefix."

        # Split prefix and patch
        P = L - self.prefix_tokens
        x_prefix = x[:, :self.prefix_tokens] if self.prefix_tokens > 0 else x.new_zeros(B, 0, D)
        x_patch0 = x[:, self.prefix_tokens:]  # [B, P, D]

        # Parse (H, W) and ensure buffers
        H, W = _norm_hw(self.user_patch_hw, P) if self.user_patch_hw is not None else _norm_hw(None, P)
        self._ensure_buffers(x.device, H, W)

        x_patch = x_patch0
        parent_bank = []
        for l, attn in enumerate(self.level_attn):
            n = int(self.sib_msk[l].sum().item())
            if n == 0: break
            idx = self.sib_idx[l, :n]            # [n,4], index 0..(H*W-1)
            child = x_patch[:, idx]              # [B, n, 4, D]
            o_parent, x_grp = attn(child)        # fixed‑k3
            
            if self.use_group_norm:
                B, n, C, D = x_grp.shape
                xg = self.group_norm(x_grp.reshape(-1, D)).view(B, n, C, D)
                parent_bank.append(xg.mean(2))      
            else:
                parent_bank.append(x_grp.mean(2))   

            x_patch = o_parent                   # [B, n, D], enter next layer

        root = self._weighted_pool(parent_bank)  # [B, D] or None
        if root is None:
            return x  # No layers to aggregate, do not change CLS

        # Replace CLS (or first prefix position) with root
        x_out = x.clone()
        if self.prefix_tokens > 0:
            x_out[:, 0, :] = root
        else:
            # No prefix, insert root at the front and discard the last patch to keep the length unchanged
            x_out = torch.cat([root.unsqueeze(1), x[:, 1:, :]], dim=1)
        return x_out
