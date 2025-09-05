![Visitor Badge](https://visitor-badge.laobi.icu/badge?page_id=Hannibal730.Webcam-Eye-Tracker)

# Webcam Eye Tracker


<img src="https://github.com/user-attachments/assets/3887fa4d-d510-4004-95ab-91bf04c8b868" width="800" alt="predict Example" />

[Youtube Link](https://youtu.be/Z9MyU2C8ZpM)

---

Low-cost eye tracking on a **standard webcam** with **simple calibration**, **monitor & camera selection**, and a transparent on-screen overlay. Built on Google **MediaPipe Face Mesh** with a compact, mathematically grounded pipeline (**eye-contour PCA → per-axis normalization → ridge regression**) for real-time inference.

> **Purpose:** make eye-based communication more accessible for people with limited mobility—using only a lower cost home webcam.


---

## Key Features

* **Google MediaPipe** (Face Mesh with iris) — robust, cross-platform landmarking.
* **Dead-simple calibration** — pick rows/cols, per-point dwell, and delay; targets auto-advance.
* **Device selection in UI** — choose the **display** used for overlay/targets and the **webcam**.
* **Mathematical pipeline** — eye-contour **PCA**, **anisotropic normalization** (separate scales for ± directions), optional **eye patches** (CLAHE + z-norm), fast **ridge regression** (dual form, **un-regularized intercept**), and **OneEuro + EMA** smoothing.
* **Clean artifacts** — models saved as `YYYYMMDD_HHMMSS_Grid{R}x{C}_Patch{W}x{H}.pkl`; datasets as `.npz`.

---

## OS / Camera / Requirements

* **OS:** Ubuntu 22.04
* **Camera tested:** Logitech **C920 Pro** (others should work)

<img src="https://github.com/user-attachments/assets/8d1d5a33-7875-47a6-ab9e-71f8cd7b2cb7" width="300" alt="predict Example" />


### Install

```bash
python -m venv .venv
source .venv/bin/activate          # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

* A **Control Panel** window appears (always on top).
* Pick **Target/Overlay monitor** and **Webcam**, set grid & timing, then press **Start Calibration**.
* After calibration, the **red dot** shows your live gaze on the chosen monitor.

---

## UI Overview

<img src="https://github.com/user-attachments/assets/362812a9-bf6a-422e-bf9c-fa724e1d4a35" width="400" alt="predict Example" />


### Display Settings

* **Target/Overlay monitor** — which display shows the black calibration background & orange targets (and the final red gaze dot).
* **Webcam** — active camera device (switches live).

### Calibration Grid

* **Rows / Columns** — grid for target positions (serpentine order).
* **Per-point (sec)** — dwell time per target.
* **Delay (sec)** — time to wait **after the target moves** before sampling starts (prevents early, off-target frames).

### Calibration Command

* **Start Calibration** — begins the target sequence and data capture.
* **Stop Calibration** — aborts the sequence (no model save).
* **Load Model (.npz/.pkl)** — load a previously saved model.
* **Hide/Show Overlay** — toggles the transparent overlay.
* **Quit** — exits the app.

### Visualization (on the preview window)

* **Iris centers / Iris 4-edges** — yellow markers for each iris.
* **Eye axes (fixed length; û, v̂)** — principal axes from PCA, constant length for reference.
* **Eye axes (eye scaled length; s\_u, s\_v)** — axes scaled by eye geometry.
* **u, v vectors / u, v vectors (bigger)** — shows current normalized offsets; the latter uses a gain.
* **Eye contour points / edges** — raw eye polygon points and wireframe.
* **Eye patch ROI boxes / Eye patch thumbnails** — oriented crop boxes and zoomed mini-patches (L/R) for debugging.

### Patch sizing

* **Height = Width ×** — vertical half-size as a ratio of horizontal half-size (keeps aspect).
* **Width scale (û RMS → half\_w)** — scales ROI width from the eye’s horizontal spread.
* **Patch width (px) / Patch height (px)** — resolution of the extracted patches (affects feature dimension).

### Smoothing Factors

* **OneEuro mincutoff / beta / dcutoff** — jitter vs. responsiveness trade-off.
* **EMA α** — exponential moving average weight (higher = smoother, slower).

---

### Calibration Guide

<img src="https://github.com/user-attachments/assets/bad50395-3ef6-4ed3-b154-20dcd180fa58" width="500" alt="predict Example" />

1. **Pick devices**
   In the Control Panel, choose the **Target/Overlay monitor** and **Webcam**.

2. **Set grid & timing**

   * **Rows / Columns**: target layout (serpentine order).
   * **Per-point (sec)**: how long to dwell on each target.
   * **Delay (sec)**: wait time after the target moves **before** sampling starts.

3. **Start**
   Click **Start Calibration**. An orange **ring** appears on a black screen. After the delay, it turns into a **filled dot**—that’s when data is collected. Keep your eyes on the dot until it jumps to the next location.

4. **Finish**
   After the last point, the model is trained and saved automatically (e.g., `YYYYMMDD_HHMMSS_Grid{R}x{C}_Patch{W}x{H}.pkl`). The overlay switches to a **red dot** showing live gaze.

5. **Controls**

   * **Stop Calibration**: aborts the sequence.
   * **Keyboard (preview window)**: `c` start, `s` stop, `o` overlay toggle, `q`/`ESC` quit.

**Tips:** keep `Delay < Per-point`, hold head steady, ensure even lighting, and avoid moving windows between monitors during calibration.


---

## Mathematical Pipeline

### 1) Landmarks & Eye Contours

Using **MediaPipe Face Mesh**, we read dense facial landmarks, including iris points.
For each eye we gather contour points $\{x_i\in\mathbb{R}^2\}$.

<br>

### 2) Eye Axes via PCA (SVD)

Compute the eye centroid $c$ and PCA on centered points $X = [x_i - c]$ to obtain unit axes:

* **û** (ax1): major principal direction
* **v̂** (ax2): minor principal direction

To avoid visual flips when head pitch changes, we fix a **sign convention** per frame:

* force **v̂ to point downward** (image $+y$)
* enforce a **right-handed frame** (if $\det[\hat u,\hat v]<0$, flip $\hat u$)

This makes patch warping and thumbnails temporally stable.

<br>

### 3) Anisotropic Per-Axis Normalization

Let the iris center be $p$. Project the offset onto the axes:

$$
\Delta u = (p-c)\cdot\hat u,\qquad \Delta v = (p-c)\cdot\hat v.
$$

From the eye contour we estimate **separate RMS scales** for the **positive** and **negative** sides along each axis:

$$
s_u^+ = \mathrm{RMS}\{t_1\ge 0\}, \quad s_u^- = \mathrm{RMS}\{t_1<0\},\quad
s_v^+ = \mathrm{RMS}\{t_2\ge 0\}, \quad s_v^- = \mathrm{RMS}\{t_2<0\}
$$

where $t_1 = (x_i-c)\cdot\hat u$, $t_2 = (x_i-c)\cdot\hat v$.

We then normalize **piecewise**

$$
\Delta u = (p-c)\cdot \hat u,\quad \Delta v = (p-c)\cdot \hat v
$$

$$
u=
\begin{cases}
\dfrac{\Delta u}{s_u^+}, & \Delta u \ge 0\\
\dfrac{\Delta u}{s_u^-}, & \Delta u < 0
\end{cases}
\qquad
v=
\begin{cases}
\dfrac{\Delta v}{s_v^+}, & \Delta v \ge 0\\
\dfrac{\Delta v}{s_v^-}, & \Delta v < 0
\end{cases}
$$

This captures eyelid asymmetry and improves vertical sensitivity.

<br>

### 4) Oriented Eye Patches (optional)

We crop an **oriented ROI** around each eye using the axes:

* Horizontal half-size: $\text{half}_w = \max(s_u^+, s_u^-)\times \text{scale}_w$
* Vertical half-size: $\text{half}_h = \text{half}_w \times \text{ratio}_{h\leftarrow w}$

We build an oriented rectangle from $(c,\hat u,\hat v,\text{half}_w,\text{half}_h)$ and **affine-warp** it to a fixed grid of size `patch_w × patch_h`.
Preprocessing: **grayscale → CLAHE → flatten → z-normalize** to a 1-D vector.

<br>

### 5) Feature Fusion

Concatenate:

* **12-D geometric**: $[u_L, v_L, u_R, v_R]$ plus quadratic/cross terms
* **Left patch vector** + **Right patch vector**

### 6) Ridge Regression (dual, intercept un-regularized)

We solve ridge in **dual form** on centered variables and recover the intercept:

$$
W = X_c^\top(K_c + \lambda I)^{-1}Y_c,\qquad
b = \bar Y - \bar X\,W
$$

This matches a **primal ridge with no penalty on $b$** but runs fast even when the feature dimension is large.

<br>

### 7) Smoothing

Final 2-D gaze is filtered with **OneEuro** and **EMA** to reduce jitter while remaining responsive.

---

## Saved Files

* **models** → `models/YYYYMMDD_HHMMSS_Grid{R}x{C}_Patch{W}x{H}.pkl`
  Includes `W`, `b`, target screen size, and feature names.
* **data** → `data/gaze_samples_YYYYMMDD_HHMMSS.npz`
  Feature matrix `X`, labels `Y`, per-target index, timestamps, and calibration meta.

---

## Notes & Tips

* **Keep lighting even and frontal for stable iris/contours.**
* Start with moderate patch sizes (e.g., `40×40`) and adjust `Width scale` + `Height ratio` for your camera/face distance.
* If you switch monitors during calibration, the sequence restarts to keep coordinates consistent.
* Preview **mirroring** affects display only (not the learned model).

---

## License

Apache License 2.0
