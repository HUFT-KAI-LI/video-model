"""Strict result joins and pinned upstream VBench aggregation, not proxy scores."""
import csv
import importlib.util
import json
import math
import runpy
import sys
from collections import Counter
from pathlib import Path

from ..io_utils import sha256, write_json
from .benchmarks import check_vendor, source_hash, validate_job_paths


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected a finite numeric score, got {value!r}')
    return float(value)


def expected_paths(jobs, videos):
    validate_job_paths(jobs)
    if not jobs:
        raise ValueError('Cannot aggregate an empty denominator')
    return {str((Path(videos)/j['relative_path']).resolve()): j for j in jobs}


def videophy2(jobs, videos, sa_path, pc_path):
    expected = expected_paths(jobs, videos)
    errors, results = [], {}
    coverage = {}
    for mode, path in [('sa', sa_path), ('pc', pc_path)]:
        matched, seen = {}, set()
        with Path(path).open(newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'videopath', 'caption', 'score'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{mode}: required VideoPhy-2 result columns are absent')
            for number, row in enumerate(reader, 2):
                identity = str(Path(row['videopath']).resolve())
                if identity not in expected or identity in seen:
                    errors.append(dict(mode=mode, row=number, reason='unexpected_or_duplicate_video', video=identity))
                    continue
                seen.add(identity)
                if row['caption'] != expected[identity]['original_prompt']:
                    errors.append(dict(mode=mode, row=number, reason='caption_mismatch', video=identity))
                    continue
                try:
                    value = finite(float(row['score']))
                    if value != int(value) or not 1 <= value <= 5:
                        raise ValueError('VideoPhy-2 requires integer ratings 1..5; 0 is a scoring failure')
                except (ValueError, TypeError, OverflowError) as error:
                    errors.append(dict(mode=mode, row=number, reason='invalid_rating', detail=str(error), video=identity))
                    continue
                matched[identity] = int(value)
        coverage[mode] = dict(expected=len(expected), valid=len(matched),
                              missing_or_invalid=sorted(set(expected)-set(matched)))
        results[mode] = matched
    complete = not errors and all(len(r) == len(expected) for r in results.values())
    report = dict(suite='videophy2', benchmark_version='VideoPhy-2', rating_scale=[1, 5],
                  denominator=len(expected), coverage=coverage, errors=errors, complete=complete,
                  scores=None, main_table_ready=complete,
                  result_sha256={'sa': sha256(sa_path), 'pc': sha256(pc_path)})
    if complete:
        pairs = [dict(job_id=j['job_id'], video=path, sa=results['sa'][path], pc=results['pc'][path],
                      joint_at_4=results['sa'][path] >= 4 and results['pc'][path] >= 4)
                 for path, j in expected.items()]
        n = len(pairs)
        report['paired_results'] = pairs
        report['scores'] = dict(sa_mean=sum(r['sa'] for r in pairs)/n,
                                pc_mean=sum(r['pc'] for r in pairs)/n,
                                sa_at_4=sum(r['sa'] >= 4 for r in pairs)/n,
                                pc_at_4=sum(r['pc'] >= 4 for r in pairs)/n,
                                joint_at_4=sum(r['joint_at_4'] for r in pairs)/n)
    return report


def official_vbench():
    repo = check_vendor('VBench')
    script, constant = repo/'scripts/cal_final_score.py', repo/'scripts/constant.py'
    provenance = {str(p.relative_to(repo)): source_hash(repo, p) for p in (script, constant)}
    spec = importlib.util.spec_from_file_location('constant', constant)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous, search_path = sys.modules.get('constant'), sys.path[:]
    try:
        sys.modules['constant'] = module
        functions = runpy.run_path(str(script))
    finally:
        sys.path[:] = search_path
        if previous is None:
            sys.modules.pop('constant', None)
        else:
            sys.modules['constant'] = previous
    return functions, provenance


def vbench(jobs, videos, result_paths, metadata=None, allow_official_color_exclusions=False):
    expected_paths(jobs, videos)
    official, sources = official_vbench()
    repo = check_vendor('VBench')
    metadata_path = repo/'vbench/VBench_full_info.json'
    sources['vbench/VBench_full_info.json'] = source_hash(repo, metadata_path)
    if metadata is None:
        metadata = json.loads(metadata_path.read_text())
    dimensions = {d.replace(' ', '_') for d in official['TASK_INFO']}
    # Derive each dimension's own denominator, retaining upstream row multiplicity.
    expected = {d: Counter(str((Path(videos)/'vbench'/f'{r["prompt_en"]}-{i}.mp4').resolve())
                           for r in metadata if d in r['dimension'] for i in range(5))
                for d in dimensions}
    planned = {str((Path(videos)/j['relative_path']).resolve()) for j in jobs}
    if planned != {p for paths in expected.values() for p in paths}:
        raise ValueError('VBench jobs do not cover the pinned standard prompt/seed filenames')
    errors, raw, coverage, provenance = [], {}, {}, {}
    for path in sorted(map(Path, result_paths)):
        provenance[str(path.resolve())] = sha256(path)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError('VBench result must be a dimension mapping')
        for dimension, result in payload.items():
            if dimension not in dimensions or dimension in coverage:
                errors.append(dict(dimension=dimension, reason='unexpected_or_duplicate_dimension'))
                continue
            if not isinstance(result, list) or len(result) != 2 or not isinstance(result[1], list):
                raise ValueError(f'Malformed official dimension result: {dimension}')
            seen, valid, invalid = Counter(), Counter(), []
            try:
                raw[dimension] = finite(result[0])
            except ValueError as error:
                errors.append(dict(dimension=dimension, reason='invalid_aggregate', detail=str(error)))
            for row in result[1]:
                path = str(Path(row['video_path']).resolve())
                seen[path] += 1
                try:
                    # Dynamic degree and human action legitimately return booleans.
                    finite(int(row['video_results']) if isinstance(row['video_results'], bool)
                           else row['video_results'])
                    valid[path] += 1
                except (KeyError, ValueError) as error:
                    invalid.append(dict(video=path, error=str(error)))
            missing = list((expected[dimension]-valid).elements())
            unexpected = list((seen-expected[dimension]).elements())
            coverage[dimension] = dict(expected=sum(expected[dimension].values()), scored=sum(valid.values()),
                                       missing_or_invalid=missing, unexpected_or_duplicate=unexpected, invalid=invalid,
                                       complete=not missing and not unexpected and not invalid)
            if dimension == 'color' and missing:
                coverage[dimension]['upstream_caveat'] = (
                    'Official color omits videos without detected objects. No zero imputation or reduced-denominator '
                    'complete score is substituted; inspect these exclusions before comparison.')
    missing_dimensions = sorted(dimensions-set(coverage))
    complete = (not errors and not missing_dimensions and set(raw) == dimensions and
                all(c['complete'] for c in coverage.values()))
    strict_coverage_complete = complete
    exclusions = {}
    if allow_official_color_exclusions and not errors and not missing_dimensions and set(raw) == dimensions:
        color = coverage.get('color', {})
        exclusions = dict(color=color.get('missing_or_invalid', []))
        # The pinned color evaluator omits no-object detections by design. Preserve
        # the scheduled denominator and expose exclusions, never zero-impute them.
        complete = all(c['complete'] or (d == 'color' and c['scored'] > 0 and
                       not c['unexpected_or_duplicate'] and not c['invalid'])
                       for d, c in coverage.items())
    report = dict(suite='vbench', benchmark_version='VBench standard', denominator=len(jobs),
                  coverage=coverage, errors=errors, missing_dimensions=missing_dimensions,
                  complete=complete, main_table_ready=complete, scores=None,
                  raw_dimensions=raw, source_sha256=sources, result_sha256=provenance)
    if allow_official_color_exclusions:
        report.update(strict_coverage_complete=strict_coverage_complete,
                      official_conditional_exclusions=exclusions,
                      exclusion_policy='Pinned official color omits videos with no detected target object; '
                          'official conditional score retained with explicit counts; no imputation')
    if complete:
        normalized = official['get_nomalized_score']({d.replace('_', ' '): v for d, v in raw.items()})
        quality = official['get_quality_score'](normalized)
        semantic = official['get_semantic_score'](normalized)
        total = official['get_final_score'](quality, semantic)
        report['scores'] = dict(quality_score=quality, semantic_score=semantic, total_score=total)
        report['normalized_weighted_dimensions'] = normalized
        report['aggregation'] = dict(quality_weight=official['QUALITY_WEIGHT'],
                                     semantic_weight=official['SEMANTIC_WEIGHT'],
                                     dimension_weights=official['DIM_WEIGHT'], clipping=False,
                                     implementation='pinned scripts/cal_final_score.py functions, unchanged')
    return report


def save_summary(report, output, model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report['model'] = model
    write_json(output/'summary.json', report)
    table = output/'summary.csv'
    if not report['complete']:
        table.unlink(missing_ok=True)
        raise ValueError('Incomplete/invalid official scores; summary.json preserves the full audit, total withheld')
    row = dict(model=model, suite=report['suite'], denominator=report['denominator'],
               evaluation_scope=report.get('evaluation_scope', 'full_benchmark'),
               full_benchmark_denominator=report.get('full_benchmark_denominator', report['denominator']),
               formal_benchmark_complete=report.get('formal_benchmark_complete', report['complete']),
               checkpoint_sha256=report.get('generation', {}).get('checkpoint_sha256'),
               jobs_sha256=report.get('jobs_sha256'), exploratory=report.get('exploratory'),
               publication_eligible=report.get('publication_eligible'), **report['scores'])
    row.update({'raw_'+k: v for k, v in report.get('raw_dimensions', {}).items()})
    row.update({'normalized_weighted_'+k.replace(' ', '_'): v
                for k, v in report.get('normalized_weighted_dimensions', {}).items()})
    with table.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return report
