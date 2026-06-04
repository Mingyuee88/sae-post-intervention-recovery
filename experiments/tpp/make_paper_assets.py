from __future__ import annotations

from math import comb
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR = Path(os.environ.get('TPP_RUN_DIR', str(Path(__file__).resolve().parents[2] / 'results' / 'sanitized')))
OUT_DIR = RUN_DIR / 'paper_assets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MS = pd.read_csv(RUN_DIR / 'recovery_mode_summary.csv')
PE = pd.read_csv(RUN_DIR / 'recovery_per_example.csv')
SCAN = pd.read_csv(RUN_DIR / 'scan_rows.csv')

DATASET_ORDER = ['bias_in_bios_set1', 'bias_in_bios_set2', 'bias_in_bios_set3', 'amazon_reviews']
DATASET_LABELS = {
    'bias_in_bios_set1': 'Bias in Bios 1',
    'bias_in_bios_set2': 'Bias in Bios 2',
    'bias_in_bios_set3': 'Bias in Bios 3',
    'amazon_reviews': 'Amazon Reviews',
}
MODE_ORDER = ['none', 'encoder']
MODE_LABELS = {'none': 'None', 'encoder': 'Encoder'}
MODE_COLORS = {'none': '#7f7f7f', 'encoder': '#0072B2'}
DATASET_COLORS = {
    'bias_in_bios_set1': '#E69F00',
    'bias_in_bios_set2': '#009E73',
    'bias_in_bios_set3': '#CC79A7',
    'amazon_reviews': '#56B4E9',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def one_sided_sign_p(n_better: int, n_total: int) -> float:
    return sum(comb(n_total, k) for k in range(n_better, n_total + 1)) / (2**n_total)


def summarize_overall() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        'recovered_valid_rate',
        'mean_reactivated_fraction_within_eligible',
        'mean_act_drift_l2',
        'mean_decode_drift_l2',
        'zero_reactivation_rate',
    ]
    overall_unweighted = MS.groupby('mode')[metrics].mean().reset_index()
    overall_unweighted.insert(0, 'group', 'All targets (unweighted)')

    weighted_rows = []
    for mode, df in PE.groupby('mode'):
        weighted_rows.append({
            'group': 'All valid examples (weighted)',
            'mode': mode,
            'recovered_valid_rate': float(df['recovered_success'].mean()),
            'mean_reactivated_fraction_within_eligible': float(df['reactivated_fraction_within_eligible'].mean()),
            'mean_act_drift_l2': float(df['final_act_drift_l2'].mean()),
            'mean_decode_drift_l2': float(df['final_decode_drift_l2'].mean()),
            'zero_reactivation_rate': float(df['zero_reactivation'].mean()),
        })
    overall_weighted = pd.DataFrame(weighted_rows)

    dataset_summary = (
        MS.groupby(['dataset_preset', 'mode'])[metrics]
        .mean()
        .reset_index()
    )
    dataset_summary['dataset_label'] = dataset_summary['dataset_preset'].map(DATASET_LABELS)

    pivot = MS.pivot_table(
        index=['dataset_preset', 'target_label_name'],
        columns='mode',
        values=metrics,
    )
    paired_delta = pd.DataFrame({
        'dataset_preset': [idx[0] for idx in pivot.index],
        'target_label_name': [idx[1] for idx in pivot.index],
        'delta_recovered_valid_rate': pivot[('recovered_valid_rate', 'encoder')] - pivot[('recovered_valid_rate', 'none')],
        'delta_reactivated_fraction': pivot[('mean_reactivated_fraction_within_eligible', 'encoder')] - pivot[('mean_reactivated_fraction_within_eligible', 'none')],
        'delta_act_drift_l2': pivot[('mean_act_drift_l2', 'encoder')] - pivot[('mean_act_drift_l2', 'none')],
        'delta_decode_drift_l2': pivot[('mean_decode_drift_l2', 'encoder')] - pivot[('mean_decode_drift_l2', 'none')],
        'delta_zero_reactivation_rate': pivot[('zero_reactivation_rate', 'encoder')] - pivot[('zero_reactivation_rate', 'none')],
    })

    best_idx = SCAN.groupby('dataset_preset')['total_metric'].idxmax()
    scan_best = SCAN.loc[best_idx, [
        'dataset_preset', 'target_label_name', 'n', 'total_metric',
        'intended_diff_only', 'avg_unintended_diff_only'
    ]].copy()
    scan_best['dataset_label'] = scan_best['dataset_preset'].map(DATASET_LABELS)

    return overall_unweighted, overall_weighted, dataset_summary, paired_delta, scan_best


def build_main_table(dataset_summary: pd.DataFrame, overall_unweighted: pd.DataFrame) -> pd.DataFrame:
    src = pd.concat([
        dataset_summary[['dataset_label', 'mode', 'recovered_valid_rate', 'mean_reactivated_fraction_within_eligible', 'mean_act_drift_l2', 'zero_reactivation_rate']].rename(columns={'dataset_label': 'row_label'}),
        overall_unweighted[['group', 'mode', 'recovered_valid_rate', 'mean_reactivated_fraction_within_eligible', 'mean_act_drift_l2', 'zero_reactivation_rate']].rename(columns={'group': 'row_label'}),
    ], ignore_index=True)
    pivot = src.pivot(index='row_label', columns='mode')
    table = pd.DataFrame({
        'row_label': pivot.index,
        'recovery_none': pivot[('recovered_valid_rate', 'none')].values,
        'recovery_encoder': pivot[('recovered_valid_rate', 'encoder')].values,
        'react_none': pivot[('mean_reactivated_fraction_within_eligible', 'none')].values,
        'react_encoder': pivot[('mean_reactivated_fraction_within_eligible', 'encoder')].values,
        'act_none': pivot[('mean_act_drift_l2', 'none')].values,
        'act_encoder': pivot[('mean_act_drift_l2', 'encoder')].values,
        'zero_none': pivot[('zero_reactivation_rate', 'none')].values,
        'zero_encoder': pivot[('zero_reactivation_rate', 'encoder')].values,
    })
    order = [DATASET_LABELS[k] for k in DATASET_ORDER] + ['All targets (unweighted)']
    table['row_label'] = pd.Categorical(table['row_label'], categories=order, ordered=True)
    table = table.sort_values('row_label').reset_index(drop=True)
    table['row_label'] = table['row_label'].astype(str)
    return table


def format_float(x: float, digits: int = 3) -> str:
    return f'{x:.{digits}f}'


def write_latex_tables(main_table: pd.DataFrame, scan_best: pd.DataFrame) -> None:
    lines = []
    lines.append('\\begin{tabular}{lcccccccc}')
    lines.append('\\toprule')
    lines.append('Dataset & Rec. None & Rec. Enc. & React. None & React. Enc. & Drift None & Drift Enc. & Zero-React None & Zero-React Enc. \\\\')
    lines.append('\\midrule')
    for _, row in main_table.iterrows():
        vals = [
            row['row_label'],
            format_float(row['recovery_none']),
            format_float(row['recovery_encoder']),
            format_float(row['react_none']),
            format_float(row['react_encoder']),
            format_float(row['act_none']),
            format_float(row['act_encoder']),
            format_float(row['zero_none']),
            format_float(row['zero_encoder']),
        ]
        lines.append(' & '.join(vals) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    (OUT_DIR / 'neurips_main_table.tex').write_text('\n'.join(lines))

    lines = []
    lines.append('\\begin{tabular}{lcccccc}')
    lines.append('\\toprule')
    lines.append('Dataset & Best target & $n$ & TPP metric & Intended diff. & Avg. unintended diff. \\\\')
    lines.append('\\midrule')
    for _, row in scan_best.sort_values('dataset_preset', key=lambda s: s.map({k:i for i,k in enumerate(DATASET_ORDER)})).iterrows():
        vals = [
            row['dataset_label'],
            str(row['target_label_name']).replace('_', '\\_'),
            str(int(row['n'])),
            format_float(row['total_metric']),
            format_float(row['intended_diff_only']),
            format_float(row['avg_unintended_diff_only']),
        ]
        lines.append(' & '.join(vals) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    (OUT_DIR / 'neurips_scan_best_table.tex').write_text('\n'.join(lines))


def plot_triptych(ms: pd.DataFrame, paired_delta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    metric_specs = [
        ('recovered_valid_rate', 'Recovery rate $\\uparrow$', False),
        ('mean_reactivated_fraction_within_eligible', 'Reactivation fraction $\\downarrow$', True),
        ('mean_act_drift_l2', 'Act drift L2 $\\downarrow$', True),
    ]

    for ax, (metric, ylabel, logy) in zip(axes, metric_specs):
        for dataset in DATASET_ORDER:
            sub = ms[ms['dataset_preset'] == dataset]
            pivot = sub.pivot(index='target_label_name', columns='mode', values=metric)
            color = DATASET_COLORS[dataset]
            for _, row in pivot.iterrows():
                ax.plot([0, 1], [row['none'], row['encoder']], color=color, alpha=0.35, lw=1.2, marker='o', ms=3)
            means = sub.groupby('mode')[metric].mean().reindex(MODE_ORDER)
            ax.plot([0, 1], means.values, color=color, lw=2.5, marker='o', ms=6, label=DATASET_LABELS[dataset])
        ax.set_xticks([0, 1], ['None', 'Encoder'])
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=0.6, alpha=0.8)

    axes[0].text(0.02, 0.98, 'encoder lowers recovery on 20/20 targets', transform=axes[0].transAxes, va='top', fontsize=7)
    axes[1].text(0.02, 0.98, 'encoder lowers reactivation on 20/20 targets', transform=axes[1].transAxes, va='top', fontsize=7)
    axes[2].text(0.02, 0.98, 'encoder lowers drift on 18/20 targets', transform=axes[2].transAxes, va='top', fontsize=7)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.06), frameon=False)
    save_fig(fig, 'neurips_main_triptych')


def plot_tradeoff(ms: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, y_metric, y_label in [
        (axes[0], 'mean_reactivated_fraction_within_eligible', 'Reactivation fraction $\\downarrow$'),
        (axes[1], 'mean_act_drift_l2', 'Act drift L2 $\\downarrow$'),
    ]:
        for dataset in DATASET_ORDER:
            sub = ms[ms['dataset_preset'] == dataset]
            pivot = sub.pivot(index='target_label_name', columns='mode', values=['recovered_valid_rate', y_metric])
            color = DATASET_COLORS[dataset]
            for _, row in pivot.iterrows():
                x0 = row[('recovered_valid_rate', 'none')]
                y0 = row[(y_metric, 'none')]
                x1 = row[('recovered_valid_rate', 'encoder')]
                y1 = row[(y_metric, 'encoder')]
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1.2, color=color, alpha=0.55))
                ax.scatter([x0], [y0], s=18, color=color, alpha=0.35, marker='o')
                ax.scatter([x1], [y1], s=20, color=color, alpha=0.85, marker='s')
            means = sub.groupby('mode')[['recovered_valid_rate', y_metric]].mean().reindex(MODE_ORDER)
            ax.plot(means['recovered_valid_rate'], means[y_metric], color=color, lw=2.2, alpha=0.9)
        ax.set_xlabel('Recovery rate $\\uparrow$')
        ax.set_ylabel(y_label)
        ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(color='#dddddd', lw=0.6, alpha=0.8)
    axes[0].text(0.03, 0.97, 'circles: none, squares: encoder', transform=axes[0].transAxes, va='top', fontsize=7)
    axes[1].text(0.03, 0.97, 'each arrow = one target class', transform=axes[1].transAxes, va='top', fontsize=7)
    save_fig(fig, 'neurips_tradeoff_scatter')


def plot_dataset_summary(ms: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.3))
    specs = [
        ('recovered_valid_rate', 'Recovery rate $\\uparrow$', False),
        ('mean_reactivated_fraction_within_eligible', 'Reactivation fraction $\\downarrow$', True),
        ('mean_act_drift_l2', 'Act drift L2 $\\downarrow$', True),
    ]
    x = np.arange(len(DATASET_ORDER))
    width = 0.18
    for ax, (metric, ylabel, logy) in zip(axes, specs):
        for j, mode in enumerate(MODE_ORDER):
            means = []
            for dataset in DATASET_ORDER:
                sub = ms[(ms['dataset_preset'] == dataset) & (ms['mode'] == mode)]
                mean = float(sub[metric].mean())
                means.append(mean)
                jitter = np.linspace(-0.04, 0.04, len(sub))
                ax.scatter(np.full(len(sub), x[DATASET_ORDER.index(dataset)] + (j-0.5)*width) + jitter, sub[metric], color=MODE_COLORS[mode], alpha=0.5, s=16)
            ax.bar(x + (j-0.5)*width, means, width=width, color=MODE_COLORS[mode], alpha=0.75, label=MODE_LABELS[mode])
        ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASET_ORDER], rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=0.6, alpha=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.07), frameon=False)
    save_fig(fig, 'neurips_dataset_summary')


def write_readme(main_table: pd.DataFrame, paired_delta: pd.DataFrame) -> None:
    pivot = MS.pivot_table(index=['dataset_preset', 'target_label_name'], columns='mode', values=['recovered_valid_rate', 'mean_reactivated_fraction_within_eligible', 'mean_act_drift_l2', 'mean_decode_drift_l2', 'zero_reactivation_rate'])
    react_better = int((pivot[('mean_reactivated_fraction_within_eligible', 'encoder')] < pivot[('mean_reactivated_fraction_within_eligible', 'none')]).sum())
    drift_better = int((pivot[('mean_act_drift_l2', 'encoder')] < pivot[('mean_act_drift_l2', 'none')]).sum())
    zero_better = int((pivot[('zero_reactivation_rate', 'encoder')] > pivot[('zero_reactivation_rate', 'none')]).sum())
    recovery_worse = int((pivot[('recovered_valid_rate', 'encoder')] < pivot[('recovered_valid_rate', 'none')]).sum())
    overall = main_table[main_table['row_label'] == 'All targets (unweighted)'].iloc[0]
    text = f'''# Layer-5 TPP Recovery Summary

Run: {RUN_DIR}

Headline findings:
- Encoder lowers reactivation on {react_better}/20 targets (one-sided sign test p={one_sided_sign_p(react_better,20):.2e}).
- Encoder raises zero-reactivation rate on {zero_better}/20 targets (p={one_sided_sign_p(zero_better,20):.2e}).
- Encoder lowers activation drift on {drift_better}/20 targets (p={one_sided_sign_p(drift_better,20):.2e}).
- Encoder lowers recovery rate on {recovery_worse}/20 targets.

Overall target-mean comparison:
- Recovery rate: {overall['recovery_none']:.3f} -> {overall['recovery_encoder']:.3f}
- Reactivation fraction: {overall['react_none']:.4f} -> {overall['react_encoder']:.4f}
- Act drift L2: {overall['act_none']:.3f} -> {overall['act_encoder']:.3f}
- Zero-reactivation rate: {overall['zero_none']:.3f} -> {overall['zero_encoder']:.3f}

Recommended figure usage:
- Main figure: neurips_tradeoff_scatter.pdf
- Alternative main figure: neurips_main_triptych.pdf
- Supplementary summary: neurips_dataset_summary.pdf
- Main table: neurips_main_table.tex
- Scan table: neurips_scan_best_table.tex
'''
    (OUT_DIR / 'README.md').write_text(text)


def main() -> None:
    overall_unweighted, overall_weighted, dataset_summary, paired_delta, scan_best = summarize_overall()
    main_table = build_main_table(dataset_summary, overall_unweighted)

    overall_unweighted.to_csv(OUT_DIR / 'overall_summary_unweighted.csv', index=False)
    overall_weighted.to_csv(OUT_DIR / 'overall_summary_weighted.csv', index=False)
    dataset_summary.to_csv(OUT_DIR / 'dataset_summary_unweighted.csv', index=False)
    paired_delta.to_csv(OUT_DIR / 'paired_target_deltas.csv', index=False)
    scan_best.to_csv(OUT_DIR / 'scan_best_targets.csv', index=False)
    main_table.to_csv(OUT_DIR / 'neurips_main_table.csv', index=False)

    write_latex_tables(main_table, scan_best)
    plot_triptych(MS, paired_delta)
    plot_tradeoff(MS)
    plot_dataset_summary(MS)
    write_readme(main_table, paired_delta)

    summary = {
        'run_dir': str(RUN_DIR),
        'output_dir': str(OUT_DIR),
        'num_targets': int(MS['target_label_name'].nunique()),
        'num_datasets': int(MS['dataset_preset'].nunique()),
        'main_table_rows': len(main_table),
    }
    (OUT_DIR / 'paper_assets_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
