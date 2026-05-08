"""Generate publication-quality thesis plots from extracted data.

Reads from thesis_plot_data/ (created by extract_plot_data.py on the cluster)
and produces 5 PDFs in plots/.

Usage (locally, after pulling repo):
    pip install mendeleev umap-learn  # one-time
    cd /path/to/errorbar_modelling
    python modelling/gnn/local_plots.py

Outputs:
    plots/energy_gnn_ensemble_regression.pdf
    plots/volume_mt_gnn_regression.pdf
    plots/feature_importance_rf_full.pdf
    plots/feature_importance_rf_precalc.pdf
    plots/gnn_element_embeddings_comparison.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None
    print("WARNING: umap-learn not installed. Embeddings comparison will skip UMAP.")

try:
    from mendeleev import element
except ImportError:
    element = None
    print("WARNING: mendeleev not installed. Embeddings will use Z numbers instead of symbols.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'thesis_plot_data')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'plots')
os.makedirs(SAVE_DIR, exist_ok=True)

# Publication rc params (ICML, sans-serif, matching gen_v2_plots.py)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # TrueType for editability
    'ps.fonttype': 42,
})

# PNG previews live alongside PDFs but in a sibling directory
PREVIEW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'plots_preview')
os.makedirs(PREVIEW_DIR, exist_ok=True)

LOG_EPS = 1e-7

# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def get_crystal_system(sg_number):
    sg = int(sg_number)
    if 1 <= sg <= 2: return "Triclinic"
    if 3 <= sg <= 15: return "Monoclinic"
    if 16 <= sg <= 74: return "Orthorhombic"
    if 75 <= sg <= 142: return "Tetragonal"
    if 143 <= sg <= 167: return "Trigonal"
    if 168 <= sg <= 194: return "Hexagonal"
    if 195 <= sg <= 230: return "Cubic"
    return "Unknown"


# ---------------------------------------------------------------------------
# Plot 1 & 2: Log-log regression scatter
# ---------------------------------------------------------------------------

def _r2_log_space(true, pred):
    """Coefficient of determination on log10(|.| + eps) values.

    For log-log parity plots this is the natural fit metric; linear-space R^2
    is dominated by the largest-magnitude points and doesn't reflect the
    visual scatter around y=x.
    """
    lt = np.log10(np.abs(true) + LOG_EPS)
    lp = np.log10(np.abs(pred) + LOG_EPS)
    ss_res = np.sum((lt - lp) ** 2)
    ss_tot = np.sum((lt - np.mean(lt)) ** 2)
    return 1 - ss_res / ss_tot


def plot_regression_log(true, pred, metrics, save_path, xlabel='True', ylabel='Predicted',
                        axis_range=(-7, 1), mae_unit=''):
    """Log10(|x| + eps) scatter with y=x reference and metric text box."""
    log_true = np.log10(np.abs(true) + LOG_EPS)
    log_pred = np.log10(np.abs(pred) + LOG_EPS)

    fig, ax = plt.subplots(figsize=(4, 4))
    # Wong-palette deep blue; alpha balanced for density readability at print size
    ax.scatter(log_true, log_pred, s=16, alpha=0.35, c='#0173B2',
               edgecolors='none', rasterized=True)

    # y = x reference (medium gray; visible but not dominant)
    lo, hi = axis_range
    ax.plot([lo, hi], [lo, hi], ls='--', c='#666666', lw=1.0, alpha=0.7, zorder=0)
    # Limits coincide with tick locations (advisor request)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

    # Major ticks at every integer; minor ticks at log10(2..9) offsets
    major_ticks = np.arange(lo, hi + 1, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    minor_offsets = np.log10(np.arange(2, 10))
    minor_ticks = np.array([n + off for n in range(int(lo), int(hi)) for off in minor_offsets])
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    # Inward ticks on all four sides (publication convention)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(which='minor', length=3, width=0.5)
    ax.tick_params(which='major', length=5, width=0.8)

    ax.set_xlabel(rf'$\log_{{10}}(|\mathrm{{{xlabel}}}| + \epsilon)$')
    ax.set_ylabel(rf'$\log_{{10}}(|\mathrm{{{ylabel}}}| + \epsilon)$')

    # Metric text box (monospace for tabular alignment)
    r2 = _r2_log_space(true, pred)
    lines = []
    if 'MAE' in metrics:
        unit_str = f' {mae_unit}' if mae_unit else ''
        lines.append(f"MAE    = {metrics['MAE']}{unit_str}")
    if 'sMAPE' in metrics:
        lines.append(f"sMAPE  = {metrics['sMAPE']}")
    if 'MagAcc' in metrics:
        lines.append(f"MagAcc = {metrics['MagAcc']}")
    lines.append(f"R²     = {r2:.3f}")
    textstr = '\n'.join(lines)
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor='#cccccc', alpha=0.9)
    ax.text(0.04, 0.96, textstr, transform=ax.transAxes,
            family='monospace', fontsize=9,
            verticalalignment='top', bbox=props)

    fig.savefig(save_path)
    # Sibling PNG for slide / preview use
    png_name = os.path.splitext(os.path.basename(save_path))[0] + '.png'
    png_path = os.path.join(PREVIEW_DIR, png_name)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved {save_path}")
    print(f"Saved {png_path}")


# ---------------------------------------------------------------------------
# Plot 3 & 4: Feature importance horizontal bars
# ---------------------------------------------------------------------------

def _clean_feature_name(name):
    """Human-readable feature names for plots."""
    name = name.replace('delta_mean_monomer_', 'Δ mean mono. ')
    name = name.replace('delta_max_monomer_', 'Δ max mono. ')
    name = name.replace('delta_min_monomer_', 'Δ min mono. ')
    name = name.replace('delta_mad_monomer_', 'Δ MAD mono. ')
    name = name.replace('delta_monomer_', 'Δ mono. ')
    name = name.replace('delta_mean_', 'Δ mean ')
    name = name.replace('original_', 'orig. ')
    name = name.replace('_per_atom', '/atom')
    name = name.replace('_', ' ')
    return name


def plot_feature_importance(csv_path, save_path, top_k=10):
    """Horizontal bar chart with error bars from permutation importance."""
    df = pd.read_csv(csv_path).sort_values('perm_importance_mean', ascending=False)
    top = df.head(top_k).iloc[::-1]  # reverse for bottom-to-top

    display_names = [_clean_feature_name(n) for n in top['feature']]

    fig, ax = plt.subplots(figsize=(5.5, 0.38 * top_k + 0.6))

    ax.barh(
        range(top_k),
        top['perm_importance_mean'],
        xerr=top['perm_importance_std'],
        color='#4878CF',
        edgecolor='#2C4F8C',
        linewidth=0.5,
        capsize=2,
        height=0.7
    )

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(display_names)
    ax.set_xlabel(r'Permutation Importance ($\Delta$sMAPE)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 5: Element embeddings — PCA / t-SNE / UMAP comparison
# ---------------------------------------------------------------------------

def _get_element_data(weights):
    """Extract valid element vectors, symbols, and periodic groups."""
    valid_vecs = []
    meta = []
    for z in range(1, len(weights)):
        vec = weights[z]
        if np.allclose(vec, 0):
            continue
        try:
            if element is not None:
                el = element(z)
                sym = el.symbol
                grp = el.group_id if el.group_id else 0
            else:
                sym = str(z)
                grp = 0
        except Exception:
            continue
        valid_vecs.append(vec)
        meta.append({'sym': sym, 'grp': grp, 'z': z})
    return np.array(valid_vecs), meta


def plot_embeddings_comparison(weights, save_path):
    """1x3 subplot: PCA, t-SNE, UMAP colored by periodic group."""
    from adjustText import adjust_text

    vecs, meta = _get_element_data(weights)
    if len(vecs) == 0:
        print("No valid embeddings found, skipping.")
        return

    groups = np.array([m['grp'] for m in meta])
    symbols = [m['sym'] for m in meta]

    # Compute projections
    pca_coords = PCA(n_components=2).fit_transform(vecs)
    tsne_coords = TSNE(n_components=2, perplexity=15, random_state=42,
                       init='pca', learning_rate='auto').fit_transform(vecs)

    projections = [('PCA', pca_coords), ('t-SNE', tsne_coords)]

    if umap is not None:
        umap_coords = umap.UMAP(n_components=2, random_state=42).fit_transform(vecs)
        projections.append(('UMAP', umap_coords))

    # Axis label suffixes per method
    axis_labels = {
        'PCA': ('PC 1', 'PC 2'),
        't-SNE': ('t-SNE 1', 't-SNE 2'),
        'UMAP': ('UMAP 1', 'UMAP 2'),
    }

    n_plots = len(projections)
    fig, axes = plt.subplots(1, n_plots, figsize=(16, 6))
    if n_plots == 1:
        axes = [axes]

    # Discrete colormap: unique groups sorted
    unique_groups = np.sort(np.unique(groups))
    n_groups = len(unique_groups)
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(n_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    colors = [cmap(group_to_idx[g]) for g in groups]

    # Only annotate a representative subset of elements
    target_labels = {'Li', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P',
                     'S', 'Cl', 'K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu',
                     'Zn', 'Pd', 'Pt', 'Au', 'Pb'}
    label_fontsize = 12

    for ax, (title, coords) in zip(axes, projections):
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=colors, s=60, edgecolors='k',
                   linewidths=0.5, alpha=0.8, zorder=2, rasterized=True)

        # Only create text objects for target elements
        texts = []
        for i, sym in enumerate(symbols):
            if sym in target_labels:
                t = ax.text(coords[i, 0], coords[i, 1], sym,
                            fontsize=label_fontsize, ha='center', va='center',
                            zorder=3)
                texts.append(t)
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        ax.set_title(title, fontsize=label_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        xl, yl = axis_labels.get(title, (f'{title} 1', f'{title} 2'))
        ax.set_xlabel(xl, fontsize=label_fontsize)
        ax.set_ylabel(yl, fontsize=label_fontsize)

    # Discrete colorbar with integer ticks — placed with manual axes to avoid overlap
    cbar_ax = fig.add_axes([0.89, 0.05, 0.018, 0.85])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=-0.5, vmax=n_groups - 0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(np.arange(n_groups))
    tick_labels = ['Ln/An' if g == 0 else str(g) for g in unique_groups]
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=label_fontsize)
    cbar_ax.set_title('Periodic\nGroup', fontsize=label_fontsize, pad=6)

    fig.subplots_adjust(left=0.04, right=0.86, wspace=0.20)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Legacy: PCA-only embeddings (backward compat with old thesis_data_export.pkl)
# ---------------------------------------------------------------------------

def plot_embeddings(weights):
    """PCA of Element Embeddings (legacy single-plot version)."""
    vecs, meta = _get_element_data(weights)
    if len(vecs) == 0:
        return

    coords = PCA(n_components=2).fit_transform(vecs)
    groups = [m['grp'] for m in meta]
    symbols = [m['sym'] for m in meta]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=groups, cmap='tab20',
                          s=150, edgecolors='k')
    for i, txt in enumerate(symbols):
        plt.annotate(txt, (coords[i, 0], coords[i, 1]),
                     xytext=(5, 5), textcoords='offset points')
    plt.colorbar(scatter, label='Group')
    plt.savefig(f"{SAVE_DIR}/embeddings_pca.png", dpi=150)
    plt.close()
    print("Saved embeddings_pca.png")


# ---------------------------------------------------------------------------
# Legacy: Violin/bar by crystal system
# ---------------------------------------------------------------------------

def plot_violin_and_bar(df, target_col, true_col, metric_name):
    """Generates Violin (Error) and Bar (Accuracy) plots."""
    import seaborn as sns

    df = df.copy()
    if metric_name == 'MAE':
        df['Metric'] = np.abs(df[target_col] - df[true_col])
        ylabel = f"MAE ({target_col})"
    elif metric_name == 'sMAPE':
        denom = np.abs(df[target_col]) + np.abs(df[true_col]) + 1e-7
        df['Metric'] = 200 * np.abs(df[target_col] - df[true_col]) / denom
        ylabel = "sMAPE (%)"
    elif metric_name == 'MagAcc':
        bins = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
        p_bins = np.digitize(np.abs(df[target_col]), bins)
        t_bins = np.digitize(np.abs(df[true_col]), bins)
        df['Metric'] = (p_bins == t_bins).astype(float)
        ylabel = "Accuracy"

    df['System'] = df['spacegroup'].apply(get_crystal_system)
    df = df[df['System'] != 'Unknown']
    counts = df['System'].value_counts()
    labels = {sys: f"{sys}\n(n={counts[sys]})" for sys in counts.index}
    df['Label'] = df['System'].map(labels)
    order = counts.index.map(labels)

    plt.figure(figsize=(10, 6))
    if metric_name == 'MagAcc':
        sns.barplot(data=df, x='Label', y='Metric', order=order, palette='viridis')
        plt.ylim(0, 1.0)
    else:
        sns.violinplot(data=df, x='Label', y='Metric', order=order, palette='viridis', cut=0)

    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{target_col}_{metric_name}.png", dpi=150)
    plt.close()
    print(f"Saved {target_col}_{metric_name}.png")


# ---------------------------------------------------------------------------
# Main: Generate all thesis plots
# ---------------------------------------------------------------------------

def main():
    print(f"Data dir: {DATA_DIR}")
    print(f"Save dir: {SAVE_DIR}\n")

    # --- Plot 1: Energy regression (ST GNN Ens Top-5) ---
    energy_csv = os.path.join(DATA_DIR, 'energy_ensemble_test.csv')
    if os.path.exists(energy_csv):
        df = pd.read_csv(energy_csv)
        plot_regression_log(
            df['energy_true'].values, df['energy_pred'].values,
            metrics={'MAE': '0.0147', 'sMAPE': '8.45%', 'MagAcc': '99.28%'},
            save_path=os.path.join(SAVE_DIR, 'energy_gnn_ensemble_regression.pdf'),
            xlabel=r'\Delta E', ylabel=r'\Delta E_{pred}',
            mae_unit='eV/atom'
        )
    else:
        print(f"SKIP: {energy_csv} not found")

    # --- Plot 2: Volume regression (MT GNN no-lattice) ---
    vol_csv = os.path.join(DATA_DIR, 'volume_mt_gnn_test.csv')
    if os.path.exists(vol_csv):
        df = pd.read_csv(vol_csv)
        plot_regression_log(
            df['volume_true'].values, df['volume_pred'].values,
            metrics={'MAE': '0.128', 'sMAPE': '24.34%', 'MagAcc': '90.61%'},
            save_path=os.path.join(SAVE_DIR, 'volume_mt_gnn_regression.pdf'),
            xlabel=r'\Delta V', ylabel=r'\Delta V_{pred}',
            mae_unit='Å³'
        )
    else:
        print(f"SKIP: {vol_csv} not found")

    # --- Plot 3: RF-Full feature importance ---
    full_csv = os.path.join(DATA_DIR, 'feature_importance_full_energy.csv')
    if os.path.exists(full_csv):
        plot_feature_importance(
            full_csv,
            save_path=os.path.join(SAVE_DIR, 'feature_importance_rf_full.pdf'),
            top_k=10
        )
    else:
        print(f"SKIP: {full_csv} not found")

    # --- Plot 4: RF-PreCalc feature importance ---
    precalc_csv = os.path.join(DATA_DIR, 'feature_importance_precalc_energy.csv')
    if os.path.exists(precalc_csv):
        plot_feature_importance(
            precalc_csv,
            save_path=os.path.join(SAVE_DIR, 'feature_importance_rf_precalc.pdf'),
            top_k=10
        )
    else:
        print(f"SKIP: {precalc_csv} not found")

    # --- Plot 5: Element embeddings comparison ---
    emb_path = os.path.join(DATA_DIR, 'z_embedding.npz')
    if os.path.exists(emb_path):
        data = np.load(emb_path)
        plot_embeddings_comparison(
            data['weights'],
            save_path=os.path.join(SAVE_DIR, 'gnn_element_embeddings_comparison.pdf')
        )
    else:
        print(f"SKIP: {emb_path} not found")

    print("\nAll thesis plots generated!")


if __name__ == "__main__":
    main()
