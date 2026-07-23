# %%
# Neural airfoil-stall demo — U-Net rollout vs lattice-Boltzmann reference
#
# Reproduces the showcase animation: a NACA 2412 airfoil at 18° angle of attack,
# deep stall, 2D unsteady flow (Re_c = 360).  Left: a D2Q9 lattice-Boltzmann
# solver computes the reference sequence.  Right: a pretrained U-Net, given only
# the first frame, generates the remaining frames autoregressively — the network
# advances the flow itself, at milliseconds per frame.
#
# This is a fixed single-case demonstration accompanying our pitch material.
# The training pipeline, datasets, and general-purpose models are not included.
#
# Requirements: python 3.10+, torch, numpy, matplotlib (ffmpeg optional, for mp4)
# Usage:        place airfoil_stepper.pt next to this file, then
#                   python stall_rollout_demo.py
# Runtime:      ~1 minute for the reference solve (GPU), seconds for the rollout.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.animation as manim
import time

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── The demo case (fixed) ─────────────────────────────────────────────────────
NACA    = [2.0, 4.0, 12.0]        # NACA 2412
AOA_DEG = 18.0                    # deep stall
NX, NY  = 400, 200
CHORD   = 120                     # pixels
U_IN    = 0.10                    # lattice units
TAU     = 0.60                    # BGK relaxation time
CS2     = 1.0/3.0
NU_LBM  = CS2*(TAU - 0.5)
RE_C    = U_IN*CHORD/NU_LBM      # chord Reynolds number = 360
N_SETTLE = 20000                  # steps past the initial transient
N_SNAP   = 60                     # frames in the animation
DT_SNAP  = 250                    # solver steps between frames
print(f'NACA 2412 @ {AOA_DEG}°   grid {NX}×{NY}   Re_c = {RE_C:.0f}')

# ══════════════════════════════════════════════════════════════════════════════
# Geometry — NACA 4-digit profile, pitched nose-up, rasterised to a solid mask
# ══════════════════════════════════════════════════════════════════════════════

def naca_polygon(params, n=300):
    """Closed polygon of a NACA 4-digit profile, chord 1, leading edge at origin."""
    x = 0.5*(1 - np.cos(np.linspace(0, np.pi, n)))
    m, p, t = params[0]/100.0, params[1]/10.0, params[2]/100.0
    yc = np.where(x < p, m/max(p**2,1e-9)*(2*p*x - x**2),
                  m/max((1-p)**2,1e-9)*((1-2*p) + 2*p*x - x**2)) if m > 0 else 0*x
    dyc = np.where(x < p, 2*m/max(p**2,1e-9)*(p - x),
                   2*m/max((1-p)**2,1e-9)*(p - x)) if m > 0 else 0*x
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return np.concatenate([np.stack([xu, yu], 1)[::-1], np.stack([xl, yl], 1)[1:]])

def airfoil_mask(naca, aoa_deg):
    """Solid mask (NY, NX): the airfoil pitched nose-up about its quarter chord;
    the oncoming flow is horizontal."""
    PXm, PYm = np.meshgrid(np.arange(NX), np.arange(NY))
    poly = naca_polygon(naca) - np.array([0.25, 0.0])
    a = np.deg2rad(aoa_deg)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    poly = poly @ R.T
    poly = poly*CHORD + np.array([0.30*NX, 0.5*NY])
    return mpath.Path(poly).contains_points(
        np.stack([PXm.ravel(), PYm.ravel()], 1)).reshape(NY, NX)

# ══════════════════════════════════════════════════════════════════════════════
# Reference solver — D2Q9 BGK lattice-Boltzmann
#   velocity inlet (Zou-He, left) / pressure outlet (right) / no-slip walls
# ══════════════════════════════════════════════════════════════════════════════

EX  = torch.tensor([0,1,0,-1,0,1,-1,-1,1], dtype=torch.float32, device=device)
EY  = torch.tensor([0,0,1,0,-1,1,1,-1,-1], dtype=torch.float32, device=device)
W9  = torch.tensor([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],
                   dtype=torch.float32, device=device)
OPP = torch.tensor([0,3,4,1,2,7,8,5,6], dtype=torch.long, device=device)
WALL = 2
jf = torch.arange(NY, dtype=torch.float32, device=device)
ii = (jf >= WALL) & (jf <= NY-1-WALL)
io = ii.clone()

def f_eq(rho, ux, uy):
    eu  = EX*ux.unsqueeze(-1) + EY*uy.unsqueeze(-1)
    usq = (ux**2 + uy**2).unsqueeze(-1)
    return W9*rho.unsqueeze(-1)*(1.0 + 3.0*eu + 4.5*eu**2 - 1.5*usq)

def lbm_step(f, solid, U_in):
    rho = f.sum(-1).clamp(0.5, 2.0)
    ux  = (f*EX).sum(-1)/rho
    uy  = (f*EY).sum(-1)/rho
    f_post = f - (f - f_eq(rho, ux, uy))/TAU
    f_s = torch.stack([torch.roll(torch.roll(f_post[:,:,i], int(EX[i]), 0),
                                  int(EY[i]), 1) for i in range(9)], dim=-1)
    s = solid.unsqueeze(-1)
    f_s = (1.0-s)*f_s + s*f_s[:, :, OPP]
    f_s[:,-1,4]=f_s[:,-1,2].clone(); f_s[:,-1,7]=f_s[:,-1,5].clone(); f_s[:,-1,8]=f_s[:,-1,6].clone()
    f_s[:, 0,2]=f_s[:, 0,4].clone(); f_s[:, 0,5]=f_s[:, 0,7].clone(); f_s[:, 0,6]=f_s[:, 0,8].clone()
    f_s[0,:,1]=f_s[0,:,3].clone(); f_s[0,:,5]=f_s[0,:,7].clone(); f_s[0,:,8]=f_s[0,:,6].clone()
    rho_in = (f_s[0,ii,0]+f_s[0,ii,2]+f_s[0,ii,4]
              + 2.0*(f_s[0,ii,3]+f_s[0,ii,6]+f_s[0,ii,7]))/(1.0-U_in)
    f_s[0,ii,1]=f_s[0,ii,3] + (2.0/3.0)*rho_in*U_in
    f_s[0,ii,5]=f_s[0,ii,7] - 0.5*(f_s[0,ii,2]-f_s[0,ii,4]) + (1.0/6.0)*rho_in*U_in
    f_s[0,ii,8]=f_s[0,ii,6] + 0.5*(f_s[0,ii,2]-f_s[0,ii,4]) + (1.0/6.0)*rho_in*U_in
    f_s[-1,:,3]=f_s[-1,:,1].clone(); f_s[-1,:,7]=f_s[-1,:,5].clone(); f_s[-1,:,6]=f_s[-1,:,8].clone()
    ux_o = -1.0 + (f_s[-1,io,0]+f_s[-1,io,2]+f_s[-1,io,4]
                   + 2.0*(f_s[-1,io,1]+f_s[-1,io,5]+f_s[-1,io,8]))
    f_s[-1,io,3]=f_s[-1,io,1] - (2.0/3.0)*ux_o
    f_s[-1,io,7]=f_s[-1,io,5] + 0.5*(f_s[-1,io,2]-f_s[-1,io,4]) - (1.0/6.0)*ux_o
    f_s[-1,io,6]=f_s[-1,io,8] - 0.5*(f_s[-1,io,2]-f_s[-1,io,4]) - (1.0/6.0)*ux_o
    return f_s

def solve_reference(solid_np):
    """Settle past the transient, then record N_SNAP frames every DT_SNAP steps."""
    solid = torch.tensor(solid_np.T.astype(np.float32), device=device)
    f = (W9.view(1,1,9)*torch.ones(NX, NY, 9, device=device)).clone()
    seq = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N_SETTLE):
            f = lbm_step(f, solid, U_IN)
        for _ in range(N_SNAP):
            for _ in range(DT_SNAP):
                f = lbm_step(f, solid, U_IN)
            rho = f.sum(-1).clamp(0.5, 2.0)
            ux = ((f*EX).sum(-1)/rho * (1.0-solid)).T.cpu().numpy()
            uy = ((f*EY).sum(-1)/rho * (1.0-solid)).T.cpu().numpy()
            seq.append(np.stack([ux, uy]))
    total = time.time() - t0
    per_frame = total/(N_SETTLE/DT_SNAP + N_SNAP)
    return np.stack(seq), per_frame

# ══════════════════════════════════════════════════════════════════════════════
# Neural stepper — U-Net that advances the flow one frame per forward pass
# ══════════════════════════════════════════════════════════════════════════════

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.SiLU())

class UNet(nn.Module):
    """Input (mask, u, v) at frame t → output (u, v) at frame t+1."""
    def __init__(self, base=32):
        super().__init__()
        self.e1 = _block(3, base)
        self.e2 = _block(base, base*2)
        self.e3 = _block(base*2, base*4)
        self.bott = _block(base*4, base*8)
        def _up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear',
                                             align_corners=False),
                                 nn.Conv2d(cin, cout, 3, padding=1))
        self.u3 = _up(base*8, base*4)
        self.d3 = _block(base*8, base*4)
        self.u2 = _up(base*4, base*2)
        self.d2 = _block(base*4, base*2)
        self.u1 = _up(base*2, base)
        self.d1 = _block(base*2, base)
        self.head = nn.Conv2d(base, 2, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
        xp = F.pad(x, (0, pw, 0, ph))
        s1 = self.e1(xp)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b  = self.bott(self.pool(s3))
        d  = self.d3(torch.cat([self.u3(b), s3], dim=1))
        d  = self.d2(torch.cat([self.u2(d), s2], dim=1))
        d  = self.d1(torch.cat([self.u1(d), s1], dim=1))
        dlt = self.head(d)[:, :, :H, :W]
        return x[:, 1:3] + dlt

# ══════════════════════════════════════════════════════════════════════════════
# Run: reference solve, then autoregressive rollout from the first frame
# ══════════════════════════════════════════════════════════════════════════════

solid = airfoil_mask(NACA, AOA_DEG)
print('Reference LBM solve...')
truth, lbm_spf = solve_reference(solid)                  # (T, 2, NY, NX)
print(f'  done ({lbm_spf:.2f} s/frame)')

net = UNet(base=32).to(device)
net.load_state_dict(torch.load('airfoil_stepper.pt', map_location=device,
                               weights_only=True))
net.eval()

mask_t = torch.from_numpy(solid.astype(np.float32)).to(device)
fluid  = (mask_t < 0.5).float()
x = torch.stack([mask_t,
                 torch.from_numpy(truth[0,0]).to(device)/U_IN,
                 torch.from_numpy(truth[0,1]).to(device)/U_IN]).unsqueeze(0)
frames = [truth[0]/U_IN]
t0 = time.time()
with torch.no_grad():
    for _ in range(N_SNAP-1):
        nxt = net(x)[0]*fluid
        frames.append(nxt.cpu().numpy())
        x = torch.cat([mask_t.unsqueeze(0), nxt]).unsqueeze(0)
nn_spf = (time.time()-t0)/(N_SNAP-1)
frames = np.stack(frames)
print(f'Neural rollout: {nn_spf*1e3:.1f} ms/frame '
      f'({lbm_spf/max(nn_spf,1e-9):,.0f}× faster than the reference solver)')

# ══════════════════════════════════════════════════════════════════════════════
# Animation — vorticity, side by side
# ══════════════════════════════════════════════════════════════════════════════

def vort(uv):
    return np.gradient(uv[1], axis=1) - np.gradient(uv[0], axis=0)

OM = 0.2   # colour scale for vorticity of the U∞-normalised fields
truth_n = truth/U_IN

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6))
ims = []
for ax, fr, ttl in [
        (a1, truth_n, f'Lattice-Boltzmann reference — {lbm_spf:.2f} s/frame'),
        (a2, frames,  f'Neural network rollout — {nn_spf*1e3:.1f} ms/frame')]:
    im = ax.imshow(np.where(solid, np.nan, vort(fr[0])), origin='lower',
                   cmap='RdBu_r', aspect='equal', vmin=-OM, vmax=OM)
    ax.contourf(solid, levels=[0.5, 2], colors='k')
    ax.set_title(ttl, fontsize=11); ax.axis('off')
    ims.append((im, fr))
fig.suptitle(f'NACA 2412 at {AOA_DEG:.0f}° — deep stall, Re_c = {RE_C:.0f} '
             f'(2D)', fontsize=12)

def _upd(t):
    for im, fr in ims:
        im.set_data(np.where(solid, np.nan, vort(fr[t])))
    return [im for im, _ in ims]

ani = manim.FuncAnimation(fig, _upd, frames=N_SNAP, blit=True)
ani.save('stall_rollout.gif', writer=manim.PillowWriter(fps=12), dpi=90)
if manim.writers.is_available('ffmpeg'):
    ani.save('stall_rollout.mp4', writer=manim.FFMpegWriter(fps=12, bitrate=6000),
             dpi=150)
print('Saved stall_rollout.gif' +
      (' and stall_rollout.mp4' if manim.writers.is_available('ffmpeg') else
       '  (install ffmpeg for mp4)'))

# %%
