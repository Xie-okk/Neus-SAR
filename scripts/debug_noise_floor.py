import argparse
import csv
import json
import os
from collections import Counter
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats


MODEL_NAMES = ('exponential', 'gamma', 'gaussian', 'lognormal', 'student_t')


def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        image = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        key = 'image' if 'image' in data else ('isar' if 'isar' in data else data.files[0])
        image = data[key]
    else:
        image = np.asarray(Image.open(path))

    image = np.asarray(image)
    original_dtype = image.dtype
    if image.ndim == 3:
        image = image[..., :3].astype(np.float64)
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

    if np.issubdtype(original_dtype, np.integer):
        image = image.astype(np.float64) / np.iinfo(original_dtype).max
    else:
        image = image.astype(np.float64, copy=False)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(image, 0.0, None)


def find_images(image_dir):
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.npy', '*.npz']
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(image_dir, pattern)))
    return sorted(paths)


def border_mask(shape, border_frac):
    if not 0.0 < border_frac < 0.5:
        raise ValueError('border_frac must be in (0, 0.5)')
    height, width = shape
    border = max(1, int(np.ceil(min(height, width) * border_frac)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask, border


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    count = int(values.size)
    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.sqrt(variance))
    median = float(np.median(values))
    eps = np.finfo(np.float64).eps

    if std > eps and count >= 4:
        skewness = float(stats.skew(values, bias=False))
        excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    return {
        'count': count,
        'mean': mean,
        'variance': variance,
        'std': std,
        'median': median,
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q90': float(np.quantile(values, 0.90)),
        'q95': float(np.quantile(values, 0.95)),
        'q99': float(np.quantile(values, 0.99)),
        'zero_ratio': float(np.mean(values == 0.0)),
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
        'variance_over_mean_sq': variance / max(mean * mean, eps),
        'mean_over_median': mean / max(median, eps),
    }


def positive_fit_data(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positive = values[values > 0.0]
    if positive.size == 0:
        return None, 0.0
    fit_epsilon = max(float(np.min(positive)) * 0.5, 1e-12)
    return np.maximum(values, fit_epsilon), fit_epsilon


def evaluate_fit(name, distribution, params, values, parameter_count):
    log_pdf = distribution.logpdf(values, *params)
    if not np.all(np.isfinite(log_pdf)):
        raise ValueError(f'{name} produced non-finite log likelihood')
    log_likelihood = float(np.sum(log_pdf))
    aic = float(2 * parameter_count - 2 * log_likelihood)
    ks_stat, ks_pvalue = stats.kstest(values, distribution.cdf, args=params)
    return {
        'name': name,
        'distribution': distribution,
        'params': tuple(float(value) for value in params),
        'log_likelihood': log_likelihood,
        'aic': aic,
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
    }


def fit_models(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positive_values, fit_epsilon = positive_fit_data(values)
    if positive_values is None or np.std(values) <= np.finfo(np.float64).eps:
        return {}, fit_epsilon

    fits = {}
    candidates = []

    exponential_params = (0.0, float(np.mean(positive_values)))
    candidates.append(('exponential', stats.expon, exponential_params, positive_values, 1))

    gamma_params = stats.gamma.fit(positive_values, floc=0.0)
    candidates.append(('gamma', stats.gamma, gamma_params, positive_values, 2))

    gaussian_params = stats.norm.fit(values)
    candidates.append(('gaussian', stats.norm, gaussian_params, values, 2))

    lognormal_params = stats.lognorm.fit(positive_values, floc=0.0)
    candidates.append(('lognormal', stats.lognorm, lognormal_params, positive_values, 2))

    student_t_params = stats.t.fit(values)
    candidates.append(('student_t', stats.t, student_t_params, values, 3))

    for name, distribution, params, fit_values, parameter_count in candidates:
        try:
            fits[name] = evaluate_fit(name, distribution, params, fit_values, parameter_count)
        except (FloatingPointError, ValueError):
            continue
    return fits, fit_epsilon


def classify_fit(fits):
    if not fits:
        return 'degenerate', 'degenerate', float('nan')
    ranked = sorted(fits.values(), key=lambda item: item['aic'])
    best_name = ranked[0]['name']
    delta_aic = float(ranked[1]['aic'] - ranked[0]['aic']) if len(ranked) > 1 else float('nan')
    family = 'heavy_tail' if best_name in ('lognormal', 'student_t') else best_name
    return best_name, family, delta_aic


def flatten_result(file_name, border_width, summary, fits, fit_epsilon):
    best_model, best_family, delta_aic = classify_fit(fits)
    row = {
        'file': file_name,
        'border_width': border_width,
        **summary,
        'fit_epsilon': fit_epsilon,
        'best_model': best_model,
        'best_family': best_family,
        'delta_aic': delta_aic,
    }
    for model_name in MODEL_NAMES:
        fit = fits.get(model_name)
        row[f'{model_name}_aic'] = fit['aic'] if fit else float('nan')
        row[f'{model_name}_ks_stat'] = fit['ks_stat'] if fit else float('nan')
        row[f'{model_name}_ks_pvalue'] = fit['ks_pvalue'] if fit else float('nan')
        row[f'{model_name}_params'] = repr(fit['params']) if fit else ''
    return row


def plot_histogram(values, fits, output_path, title, bins, hist_quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    x_max = float(np.quantile(values, hist_quantile))
    if x_max <= 0.0:
        x_max = max(float(np.max(values)), 1.0)

    edges = np.linspace(0.0, x_max, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / max(values.size, 1) / widths
    centers = 0.5 * (edges[:-1] + edges[1:])
    x = np.linspace(0.0, x_max, 800)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis in axes:
        axis.bar(centers, density, width=widths, alpha=0.45, color='tab:blue', label='background')
        for model_name, fit in fits.items():
            pdf_x = np.maximum(x, 1e-12) if model_name in ('exponential', 'gamma', 'lognormal') else x
            pdf = fit['distribution'].pdf(pdf_x, *fit['params'])
            pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
            axis.plot(x, pdf, linewidth=1.4, label=f"{model_name} AIC={fit['aic']:.1f}")
        axis.set_xlim(0.0, x_max)
        axis.set_xlabel('Normalized power')
        axis.set_ylabel('Probability density')
        axis.grid(True, alpha=0.25)

    axes[0].set_title('Linear density')
    axes[1].set_title('Log-density tail view')
    axes[1].set_yscale('log')
    positive_density = density[density > 0.0]
    if positive_density.size > 0:
        axes[1].set_ylim(max(float(np.min(positive_density)) * 0.2, 1e-8), None)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_values(file_name, values, border_width):
    summary = describe(values)
    fits, fit_epsilon = fit_models(values)
    row = flatten_result(file_name, border_width, summary, fits, fit_epsilon)
    return row, fits


def main():
    parser = argparse.ArgumentParser(description='Fit noise distributions to the outer border of radar power images.')
    parser.add_argument('--image_dir', type=str, default=os.path.join('public_data', '1999JV6', 'image'))
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help=(
            'Output directory. By default, write next to the input image '
            'directory as public_data/<case>/noise_distribution.'
        ),
    )
    parser.add_argument('--border_frac', type=float, default=0.10)
    parser.add_argument('--bins', type=int, default=80)
    parser.add_argument('--hist_quantile', type=float, default=0.999)
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        case_dir = os.path.dirname(os.path.normpath(args.image_dir))
        args.out_dir = os.path.join(case_dir, 'noise_distribution')

    paths = find_images(args.image_dir)
    if args.max_images is not None:
        paths = paths[:args.max_images]
    if not paths:
        raise FileNotFoundError(f'No images found under {args.image_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    hist_dir = os.path.join(args.out_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)

    rows = []
    normalized_backgrounds = []
    best_model_counts = Counter()

    for path in paths:
        image = load_image(path)
        mask, border_width = border_mask(image.shape, args.border_frac)
        background = image[mask]
        row, fits = analyze_values(os.path.basename(path), background, border_width)
        rows.append(row)
        best_model_counts[row['best_model']] += 1

        if row['mean'] > 0.0:
            normalized_backgrounds.append(background / row['mean'])

        histogram_path = os.path.join(hist_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
        plot_histogram(
            background,
            fits,
            histogram_path,
            f"{os.path.basename(path)} border={args.border_frac:.0%}, best={row['best_model']}",
            args.bins,
            args.hist_quantile,
        )
        print(
            f"{row['file']}: mean={row['mean']:.6g} var={row['variance']:.6g} "
            f"median={row['median']:.6g} skew={row['skewness']:.3f} "
            f"kurt={row['excess_kurtosis']:.3f} best={row['best_model']}"
        )

    if normalized_backgrounds:
        pooled = np.concatenate(normalized_backgrounds)
        pooled_row, pooled_fits = analyze_values('__ALL_NORMALIZED__', pooled, -1)
        rows.append(pooled_row)
        plot_histogram(
            pooled,
            pooled_fits,
            os.path.join(args.out_dir, 'all_normalized_backgrounds.png'),
            f"All borders normalized by per-image mean, best={pooled_row['best_model']}",
            args.bins,
            args.hist_quantile,
        )
    else:
        pooled_row = None

    csv_path = os.path.join(args.out_dir, 'noise_distribution.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'image_dir': os.path.abspath(args.image_dir),
        'image_count': len(paths),
        'border_fraction': args.border_frac,
        'best_model_counts': dict(best_model_counts),
        'all_normalized_best_model': pooled_row['best_model'] if pooled_row else None,
        'all_normalized_best_family': pooled_row['best_family'] if pooled_row else None,
        'notes': {
            'exponential_reference': {
                'variance_over_mean_sq': 1.0,
                'mean_over_median': float(1.0 / np.log(2.0)),
                'skewness': 2.0,
                'excess_kurtosis': 6.0,
            },
            'gamma_effective_looks': 'L_eff = mean^2 / variance = 1 / variance_over_mean_sq',
            'ks_pvalue': 'Approximate only because distribution parameters are fitted from the same samples.',
            'pooled_data': 'Each image border is divided by its own mean before pooling to avoid mixing different noise scales.',
        },
    }
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f'CSV saved to {csv_path}')
    print(f'Summary saved to {summary_path}')
    print(f'Histograms saved to {hist_dir}')


if __name__ == '__main__':
    main()
