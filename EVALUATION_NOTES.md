# Evaluation notes & technical caveats

This file explains the provenance of the numerical values in `README.md`. The values below were **read from the supplied saved notebook results** and were **not independently reproduced** for portfolio preparation.

## Results and provenance

- **Experiment 1 (2D VAE)**: Fashion-MNIST unsupervised training used **70,000 training + test images** combined, with **30 epochs**. Recorded final reconstruction loss approximately **255.09**. Qualitative observations indicate two latent dimensions were active but not enough for clear separation among all ten classes.
- **Experiments 2–4**: The quick comparison tested `latent_dim = 2, 3, 4, 5` for **10 epochs** on a balanced **20,000-image** training subset. `latent_dim=4` had the lowest reconstruction loss reported in this quick comparison (~251.32). `latent_dim=3` (~266.78) was retained for interpretability and dimension activity, **not** selected as minimum-loss.
- **Experiment 5**: Improved VAE with `latent_dim=3` trained for **30 epochs** on a dataset combining Fashion-MNIST train and test images; training reconstruction loss ended at ~238.82. On the common set of 10,000 images, the notebook recorded baseline mean summed image BCE **255.9446**, improved mean summed image BCE **238.8934**, a **6.66% reduction**. A companion paired-error calculation recorded **94.62%** of images with lower BCE for the improved model. Differences in rounding come from distinct output cells.
- **Class-centre generation**: Ten prototypes are decoded from per-label averages of encoded vectors. This is label-guided and does not establish unbiased random sampling performance.
- **Autoencoder comparison**: The deterministic AE used a 3D bottleneck with ~206,020 trainable parameters. Its averaged BCE training/validation loss is on a **different scale** from summed VAE BCE plus KL, so raw loss numbers should not be directly compared.

## Critical limitations

1. **The VAE experiments combined the standard train and test splits**. Their reconstruction comparison is *not* an unbiased, unseen-test evaluation. Claims should be phrased as historical paired reconstruction results, not test accuracy or generalisation.
2. **Single-run model/latent comparisons** can vary across random seeds, initialisation, sampling and training order. The heuristic `std(z_mean) > 0.2` used to label dimensions active is not a universal definition of posterior collapse.
3. **Different model capacities**: The improved VAE has more parameters than the baseline. The comparison isolates no single causal architectural change and should not be presented as strictly capacity-controlled.
4. **No saved model weights in original uploaded ZIP.** Notebook 03 expects baseline weight files created by Notebook 02. Executing a notebook in isolation may therefore fail; run in the documented order.
5. **No independent environment execution** was performed during packaging. Code was parsed for Python syntax and the saved figures were checked for valid PNG data; TensorFlow training and full notebook runs have not been verified.
6. **No DCGAN included**: The source ZIP contains only Fashion-MNIST VAE notebooks and related analysis. Do not represent this repository as a GAN project.
7. Academic integrity and permissions: Confirm the right to publish assessment-derived notebooks and figures. Do not upload classroom-only documents, original lecture templates, or restricted datasets without permission.

## Suggestions for a more rigorous follow-up

- Train each candidate model strictly on Fashion-MNIST's **60,000-image training split**, create a **validation subset only from those training images**, and evaluate reconstructed images on the untouched **10,000-image test split**.
- Repeat each latent setting with at least **three to five independent random seeds** and report mean and standard deviation.
- Compare models using consistent reconstruction reduction, KL scale and capacity controls where appropriate; consider FID/KID only with suitable reference set and interpretation.
- Export the model architecture/configuration and encoder/decoder weights for repeatable inference after retraining.
