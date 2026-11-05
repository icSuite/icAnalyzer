"""Compare current annotated retrieval statistics with saved conditional Kp ensembles."""

#%% Inputs: freeze selection, reuse ingestion, never alter the model or live reviews
import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from lompe.utils.conductance import hardy
from icanalyzer.conductance import load_conductance_cs_files
from icanalyzer.wic_si12.data import WICSI12ArrayDataset
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.manual_masking.annotated_data import load_annotations, select_annotation_files, annotation_intervals, apply_annotations
from scripts.IMAGE.vae_pipeline.run_wic_si12_pipeline import discover_files


def summarize(values, valid, quality, orbit, kp, intervals, support, minimum_frames, minimum_orbits):
    """Neutral pixel statistics, coverage and orbit/quality sensitivity by activity."""
    rows, saved = [], {}
    for k in range(10):
        selected = (kp >= k) & (kp < k + 1)
        mask = valid[selected] & support[None, None]
        count = mask[:, 0].sum(axis=0)
        orbits = np.unique(orbit[selected])
        orbit_count = np.zeros(support.shape, dtype=int)
        orbit_maps = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            x = np.where(mask, values[selected], np.nan)
            if selected.any():
                maps = np.stack([np.nanmean(x, axis=0), *np.nanquantile(x, [.1, .5, .9], axis=0)])
                for number in orbits:
                    cells = mask[orbit[selected] == number]
                    orbit_count += cells[:, 0].any(axis=0)
                    orbit_maps.append(np.nanmean(x[orbit[selected] == number], axis=0))
                orbit_mean = np.nanmean(orbit_maps, axis=0)
                weights = np.where(mask, quality[selected], 0)
                total_weight = weights.sum(axis=0)
                numerator = (weights * np.where(mask, np.log10(np.maximum(values[selected], 0) + 1), 0)).sum(axis=0)
                log_mean = np.full(numerator.shape, np.nan)
                np.divide(numerator, total_weight, out=log_mean, where=total_weight > 0)
                log_mean = 10**log_mean - 1
            else:
                maps = np.full((4, 2, *support.shape), np.nan)
                orbit_mean = log_mean = np.full((2, *support.shape), np.nan)
        reliable = support & (count >= minimum_frames) & (orbit_count >= minimum_orbits)
        finite_quality = quality[selected][:, 0][mask[:, 0]]
        row = {
            'Kp_bin': k, 'frames': int(selected.sum()), 'orbits': len(orbits),
            'unique_three_hour_intervals': len(np.unique(intervals[selected])),
            'observed_Kp_range': [float(kp[selected].min()), float(kp[selected].max())] if selected.any() else None,
            'observed_Kp_median': float(np.median(kp[selected])) if selected.any() else None,
            'reliable_cells': int(reliable.sum()),
            'quality_quantiles_p10_p50_p90': np.quantile(finite_quality, [.1, .5, .9]).tolist() if len(finite_quality) else None
        }
        rows.append(row)
        saved[k] = dict(maps=maps, orbit_mean=orbit_mean, weighted_log_mean=log_mean, count=count, orbit_count=orbit_count, reliable=reliable)
    # Kp0 has no in-range model counterpart; match the spatial domain for bins1–6.
    common = support.copy()
    for k in range(1, 7):
        common &= saved[k]['reliable']
    for row in rows:
        k = row['Kp_bin']
        for name, cells in [('bin_support', saved[k]['reliable']), ('common_support', common & saved[k]['reliable'])]:
            row[name] = {'cells': int(cells.sum())}
            for label, field in [('mean', saved[k]['maps'][0]), ('q10', saved[k]['maps'][1]), ('q50', saved[k]['maps'][2]), ('q90', saved[k]['maps'][3]), ('orbit_balanced_mean', saved[k]['orbit_mean']), ('quality_weighted_log_mean', saved[k]['weighted_log_mean'])]:
                row[name][label + '_map_max_H_P_S'] = np.max(field[:, cells], axis=1).tolist() if cells.any() else None
    return rows, saved, common


def plot_comparison(maps, priors, hardys, comparison_support, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for c, field in enumerate(['Hall', 'Pedersen']):
        for j, (statistic, map_index) in enumerate([('mean', 0), ('q50', 2)]):
            axis = axes[c, j]
            for name, label in [('hybrid_all', 'Current hybrid retrievals'), ('JSON_reviewed_only', 'Current JSON retrievals'), ('original_training_orbits_current_masks', 'Original train orbits, current masks')]:
                y = [np.max(maps[name][k]['maps'][map_index, c][comparison_support]) if comparison_support.any() else np.nan for k in range(1, 7)]
                axis.plot(range(1, 7), y, 'o-', label=label)
            if comparison_support.any():
                axis.plot(range(1, 7), [np.max(priors[k][map_index, c][comparison_support]) for k in range(1, 7)], 's--', label='Saved conditional prior')
                axis.plot(range(1, 7), [np.max(hardys[k][c][comparison_support]) for k in range(1, 7)], '^--', label='Lompe Hardy auroral')
            axis.set(title=f'{field}: maximum of {statistic} map', xlabel='Retrieval Kp bin [k,k+1) / model exact k', ylabel='S', xticks=range(1, 7))
            axis.legend(fontsize=7)
    figure.suptitle(f'Current annotations: identical {comparison_support.sum()} covered cells across all populations and bins1–6')
    figure.savefig(output / 'conductance_vs_kp.png', dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for c, field in enumerate(['Hall', 'Pedersen']):
        for index, label, color in [(1, '10th percentile', 'tab:blue'), (2, 'median', 'tab:orange'), (3, '90th percentile', 'tab:green')]:
            if not comparison_support.any():
                continue
            raw = [np.max(maps['hybrid_all'][k]['maps'][index, c][comparison_support]) for k in range(1, 7)]
            prior = [np.max(priors[k][index, c][comparison_support]) for k in range(1, 7)]
            axes[c].plot(range(1, 7), raw, 'o-', color=color, label='Retrieval ' + label)
            axes[c].plot(range(1, 7), prior, 's--', color=color, label='Prior ' + label)
        axes[c].set(title=field, xlabel='Retrieval Kp bin [k,k+1) / model exact k', ylabel='Quantile-map maximum (S)', xticks=range(1, 7))
        axes[c].legend(fontsize=8)
    figure.savefig(output / 'quantiles_vs_kp.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--annotations', type=Path, default=Path('data/partial_image_annotations_46x46.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-frames', type=int, default=20)
    parser.add_argument('--minimum-orbits', type=int, default=3)
    args = parser.parse_args()
    if args.minimum_frames < 1 or args.minimum_orbits < 1:
        raise ValueError('coverage thresholds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = args.run_directory.expanduser() / 'conductance_vae.pt'
    model_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, transform, artifact = load_run_artifact(checkpoint)
    saved_reviews = artifact['data_provenance']['manual_annotations']
    if not saved_reviews or 'csv_fallback' not in saved_reviews or model.kp_conditioning is None:
        raise ValueError('requires the native conditional hybrid run')
    snapshot = args.annotations.read_bytes()
    (args.output / 'annotations_snapshot.json').write_bytes(snapshot)
    segments = Path(saved_reviews['csv_fallback']['filename'])
    (args.output / 'segments_snapshot.csv').write_bytes(segments.read_bytes())
    annotations = load_annotations(args.output / 'annotations_snapshot.json', args.output / 'segments_snapshot.csv')
    source_dirs = sorted({str(Path(path).parent) for paths in artifact['split_files'].values() for path in paths})
    files = select_annotation_files(discover_files(source_dirs, 'conductance_cs'), annotations)
    print(f'Loading {len(files)} current hybrid source orbits', flush=True)
    product = load_conductance_cs_files(files, wic_source_intervals=annotation_intervals(annotations), include_kp=True)
    sources = product['metadata']
    product = apply_annotations(product, annotations)
    dataset = WICSI12ArrayDataset(product, transform, maximum_missing_fraction=artifact['training_configuration']['maximum_missing_fraction'])
    indices = dataset.selected_indices
    orbit, intervals = product['orbit'][indices], product['kp_interval_start'][indices]
    classes = product['annotation_class'][indices]
    support = np.asarray(artifact['data_provenance']['training_support'], dtype=bool)
    grid = artifact['data_provenance']['grid']
    if any(not np.allclose(product['grid'][name], grid[name]) for name in ('mlat', 'mlt')):
        raise ValueError('current source and saved model grids differ')
    populations = {'hybrid_all': np.ones(len(dataset), dtype=bool), 'JSON_reviewed_only': classes != 'csv', 'original_training_orbits_current_masks': np.isin(orbit, [int(Path(p).stem.split('_')[-1]) for p in artifact['split_files']['train']])}
    reports, maps, domains, arrays = {}, {}, {}, {}
    for name, selected in populations.items():
        rows, fields, common = summarize(dataset.values[selected], dataset.common_valid[selected], dataset.quality[selected], orbit[selected], dataset.kp[selected], intervals[selected], support, args.minimum_frames, args.minimum_orbits)
        reports[name], maps[name], domains[name] = rows, fields, common
        for k, field in fields.items():
            for label, array in field.items():
                arrays[f'{name}_kp{k}_{label}'] = array
        arrays[name + '_common_support'] = common
        print(name, 'frames:', int(selected.sum()), 'common cells:', int(common.sum()), flush=True)
    population_counts = {}
    for name, selected in populations.items():
        population_counts[name] = {
            'retained_frames': int(selected.sum()),
            'frames_Kp_below7': int(np.sum(selected & (dataset.kp < 7))),
            'frames_Kp_at_least7': int(np.sum(selected & (dataset.kp >= 7))),
            'Kp_range': [float(dataset.kp[selected].min()), float(dataset.kp[selected].max())]
        }
    priors, hardys = {}, {}
    for k, directory in enumerate(sorted((args.run_directory.expanduser() / 'conductance_kp_0_6').glob('kp_*'))):
        metadata = json.loads((directory / 'sampling.json').read_text())
        if metadata['Kp'] != k or metadata['model_sha256'] != model_sha:
            raise ValueError('saved sampler must contain Kp0–6 from this checkpoint')
        with np.load(directory / 'ensemble_products.npz') as archive:
            if not np.array_equal(archive['conductance_supported'], support):
                raise ValueError('ensemble support mismatch')
            priors[k] = np.stack([archive[key] for key in ['conductance_mean', 'conductance_q10', 'conductance_q50', 'conductance_q90']])
        grid = artifact['data_provenance']['grid']
        hardys[k] = np.stack(hardy(np.asarray(grid['mlat']), np.asarray(grid['mlt']), k))
        arrays[f'prior_kp{k}'], arrays[f'hardy_kp{k}'] = priors[k], hardys[k]
        for name, rows in reports.items():
            for tag, mask in [('bin_support', maps[name][k]['reliable']), ('common_support', domains[name] & maps[name][k]['reliable'])]:
                rows[k][tag]['prior_mean_map_max_H_P_S'] = np.max(priors[k][0][:, mask], axis=1).tolist() if mask.any() else None
                rows[k][tag]['prior_q50_map_max_H_P_S'] = np.max(priors[k][2][:, mask], axis=1).tolist() if mask.any() else None
                rows[k][tag]['Hardy_map_max_H_P_S'] = np.max(hardys[k][:, mask], axis=1).tolist() if mask.any() else None
    comparison_support = np.logical_and.reduce(list(domains.values()))
    arrays['cross_population_comparison_support'] = comparison_support
    if len(priors) != 7:
        raise ValueError('need all seven saved Kp ensembles')
    #%% Reproducible small arrays, frame identities and explicit qualifications
    with (args.output / 'retained_frames.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['orbit', 'time_UTC', 'Kp', 'Kp_interval_start_UTC', 'annotation_class', 'observed_supported_pixels'])
        writer.writerows(zip(orbit, product['time'][indices].astype(str), dataset.kp, intervals.astype(str), classes, (dataset.common_valid[:, 0] & support).sum(axis=(1, 2))))
    np.savez_compressed(args.output / 'statistic_maps.npz', **arrays)
    qualifications = [
        'New annotation selection, not reconstructed historical training masks.',
        'All retained Kp0–9 images are summarized; comparison with saved priors/Hardy uses bins0–6. Counts at Kp>=7 are explicit and outside original training coverage.',
        'Neutral physical mean/quantiles give equal weight to valid observed frames at each pixel; missing cells are never filled with zero.',
        'Orbit-balanced means first average each orbit at each pixel, then equally average contributing orbits.',
        'Quality-weighted log mean is inverse of weighted mean log10(S+1), not the physical arithmetic mean.',
        'Bins [k,k+1) contain fractional Kp; saved prior and Hardy are evaluated at exact integer k, so comparisons are approximate.',
        'Common support requires minimum counts and orbits in every bin1–6; bin0 uses its own reliable intersection and generated0 is extrapolation.',
        'Pixel availability and manual selection can remain activity dependent; frame/orbit counts do not ensure population representativeness.',
        'Hardy is Lompe auroral hardy(MLAT,MLT,Kp), without EUV or added background; component differences with native retrievals can remain.'
    ]
    report = {
        'model_sha256': model_sha,
        'current_annotation_provenance': annotations['provenance'],
        'historical_annotation_sha256': saved_reviews['sha256'],
        'source_products': sources, 'annotation_selection': product['annotation_selection'],
        'intrinsic_missing_fraction_cutoff': dataset.maximum_missing_fraction,
        'coverage_thresholds': dict(frames=args.minimum_frames, orbits=args.minimum_orbits),
        'populations': reports, 'population_counts': population_counts,
        'cross_population_common_cells': int(comparison_support.sum()),
        'common_cells': {name: int(domain.sum()) for name, domain in domains.items()},
        'qualifications': qualifications
    }
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    #%% Fixed spatial domain and distinct mean-versus-median comparisons
    plot_comparison(maps, priors, hardys, comparison_support, args.output)
    for source in sources:
        stat = Path(source['filename']).stat()
        if stat.st_size != source['file_size_bytes'] or stat.st_mtime_ns != source['file_mtime_ns']:
            raise RuntimeError('source product changed during analysis')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model_sha:
        raise RuntimeError('checkpoint changed during analysis')
    print('Output:', args.output, flush=True)


if __name__ == '__main__':
    main()
