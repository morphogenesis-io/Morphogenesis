# Neural Airfoil-Stall Rollout — Demo

A pretrained neural network simulates deep stall of a NACA 2412 airfoil at 18°
angle of attack. Given only the first flow frame, the U-Net advances the flow
autoregressively — vortex by vortex — at **milliseconds per frame**, next to a
lattice-Boltzmann reference solver computing the same case.

![demo](stall_rollout.gif)

## What this is

- A **fixed, single-case demonstration** accompanying our pitch material.
- 2D unsteady flow, chord Reynolds number 360. The physics regime is stated
  honestly: this is not flight-Reynolds aerodynamics — it demonstrates that a
  network can *learn the time-step map of a solver* and reproduce unsteady
  separation dynamics at a fraction of the cost.
- The network was **not trained on this airfoil/angle combination** — it is a
  held-out case.
- Training combined supervised next-frame prediction with **physics
  regularisation** (an incompressibility penalty on the predicted fields) —
  the model is constrained toward mass-conserving flow, not just
  pixel-matching.

## What this is not

The training pipeline, dataset generation, multi-airfoil/multi-angle models,
and our steady-RANS branch (trained on the
[AirfRANS](https://airfrans.readthedocs.io/) dataset, ODbL-1.0) are **not part
of this release**.

## Run it

```bash
pip install torch numpy matplotlib
python stall_rollout_demo.py        # needs airfoil_stepper.pt in this folder
```

~1 minute for the reference solve on a GPU, seconds for the neural rollout.
Outputs `stall_rollout.gif` (and `.mp4` if ffmpeg is installed).

## License

All rights reserved. Provided for evaluation and demonstration purposes;
contact us for any other use.
