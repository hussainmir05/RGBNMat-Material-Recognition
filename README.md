# Surface Material Recognition via RGBD Camera-Based Diffuse and Specular Reflectance Estimation

Paper resources and reference implementation of the reflectance decomposition described in *"Surface Material Recognition via RGBD Camera-Based Diffuse and Specular Reflectance Estimation"* (IEEE SENSORS 2026).

Khadim Hussain<sup>1</sup>, Sukhan Lee<sup>2</sup>, Eunil Park<sup>1</sup>
<sup>1</sup>Sungkyunkwan University, Suwon, Republic of Korea &nbsp;·&nbsp; <sup>2</sup>Sejong University, Seoul, Republic of Korea


---

## Overview

RGB-only material recognition breaks down when materials share color and texture. Adding NIR, polarization, or hyperspectral sensing helps, but each costs extra hardware. This work shows that the depth sensor a robot already carries is enough: structured-light point clouds are decomposed into per-point diffuse and specular reflectance using the Ward BRDF model, and those maps are fused with RGB in a two-branch EfficientNet-B0 classifier.

<p align="center">
  <img src="/methodology (1).png" width="700" alt="Framework overview">
</p>

**Key findings**

- +9.77 percentage points over the RGB baseline under leave-one-object-out cross-validation (74.66% → 84.43%).
- 86.76% on a fully held-out set of unseen objects, against 80.37% for RGB alone.
- Ward roughness α is estimated from the local normal field rather than hand-tuned.
- Decomposition is closed-form and per-point: 87.6 ms per 32×32 ROI on an Intel i7-14700F with an RTX 4060 Ti.

## Method

Reflected intensity at a surface point is modeled with the Ward BRDF:

$$I(p) = k_d\,\rho_d(p) + k_s\,\rho_s(p)$$

$$\rho_d(p) = \frac{\cos\theta_i(p)}{\pi}, \qquad
\rho_s(p) = \frac{\cos\theta_i(p)\,\exp\!\left(-\tan^2\theta_h(p)/\alpha^2\right)}{4\pi\alpha^2\sqrt{\cos\theta_i(p)\cos\theta_v(p)}}$$

The incident, viewing, and half-angle terms come from surface normals estimated on the reconstructed point cloud, combined with the calibrated camera–projector geometry. Roughness is derived from the spread of local normals:
$$\alpha = \sqrt{\frac{\mathrm{std}(-n_x/n_z)^2 + \mathrm{std}(-n_y/n_z)^2}{2}}$$

This yields small α for smooth surfaces such as ceramic and metal, larger α for fabric and stone. Because it is measured from a captured point cloud, it reflects macroscopic rather than microscopic roughness.

Coefficients are recovered two ways. **Single-view** assumes `k_d = 1 − k_s`, reducing the model to one unknown solvable from a single image; convenient, though the assumption weakens for strongly specular materials such as metal and vinyl. **Multi-view** captures the same point from two turntable poses 15° apart, giving two equations and solving for `k_d` and `k_s` with no prior constraint.

The resulting maps feed a two-branch network: one branch takes RGB, the other takes stacked diffuse and specular. Both use EfficientNet-B0 (5.3M parameters, selected over ResNet and MobileNetV2 under identical training). Classification heads are replaced by fully connected layers with batch normalization, Leaky ReLU, and dropout; features are concatenated for the final prediction. Training used Adam, learning rate 1e-4, batch size 32, 80 epochs, 224×224 inputs.

## Results

**Leave-one-object-out cross-validation**

| Material | RGB (%) | RGB + DS, single-view (%) | RGB + DS, multi-view (%) |
|---|---|---|---|
| Ceramic | 91.88 | 99.30 | 99.30 |
| Fabric  | 77.24 | 81.05 | 84.00 |
| Metal   | 63.57 | 66.71 | 74.71 |
| Paper   | 60.32 | 66.81 | 68.14 |
| Stone   | 89.99 | 98.80 | 98.80 |
| Vinyl   | 64.96 | 81.60 | 81.60 |
| **Average** | **74.66** | **82.38** | **84.43** |

**Held-out test set (18 unseen objects, 3 per category)**

| Input setting | Accuracy (%) |
|---|---|
| RGB | 80.37 |
| RGB + Roughness (geometry baseline) | 81.97 |
| RGB + DS, single-view (`k_d = 1 − k_s`) | **86.76** |
| RGB + DS, multi-view | 86.11 |

Single- and multi-view differ by 0.65 pp on only 18 objects, within expected variance. The cross-validation numbers over the full object set are the more stable comparison, and there multi-view is consistently ahead.

## What is in this repository

| | |
|---|---|
| `scripts/` | Reference implementation of the reflectance decomposition: geometric terms, roughness estimation, single- and multi-view solves |
| `config/` | Object IDs for each cross-validation fold and for the held-out set |
| `model/` | Trained two-branch checkpoints |
| `assets/` | Figures used in this README |


## What is not included

Data capture and preprocessing depend on lab-specific hardware — a structured-light 3D camera, projector calibration, turntable control, and the automatic exposure strategy for reflective surfaces described in the cited prior work — so those components are not released. The paper documents the acquisition procedure in enough detail to reimplement it on comparable equipment, and the decomposition code here is independent of the capture rig: it takes a point cloud, an image, and calibration parameters.

## Dataset

Six material categories — ceramic, metal, stone, vinyl, fabric, and paper — each with ten object instances, captured on a turntable at 15° intervals with a structured-light 3D camera. Ambient illumination is removed by subtracting an ambient-light capture. All crops are 224×224. A further set of objects, three per category, is held out from every training stage for the generalization test.

The full dataset is available from the authors upon reasonable request. A small sample sufficient to run the demo is included in this repository.



## Acknowledgments

Supported in part by the "Intelligent Manufacturing Solution under Edge-Brain Framework" project of IITP under Grant IITP-2022-0-00067 (EdgeBrain-2), sponsored by the Korea Ministry of Science and ICT, and in part by the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (RS-2026-25556365).

Built on [Open3D](https://github.com/isl-org/Open3D) and [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).


## Contact

Khadim Hussain — hussainmir05@yahoo.com
Questions are welcome through the GitHub issue tracker.
