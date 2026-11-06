"""Annotate withdrawn historical-shortlist results without rewriting raw evidence."""
import argparse
import json
from pathlib import Path

RUNS = '''small_official_20261010 adajepa_phyre_20261010 matched_base_q0_20261012
milestone_memory_20261015 milestone_scene_baseline_20261015 milestone_oracle_latent_20261015
milestone_baselines_20261015 deploy_pool_updater_20261016
memory_deploy_pilot_20260929 memory_deploy_score_diagnostic_20260929 memory_redesign_20260928
stable_scorer_cross_20260930 attribution_stage_a_cross_20260930
probe_adaptation_cross_20261001 adapted_probe_closed_loop_20261001
support_utility_20261004 support_within_20261004 memory_fresh_geometry_20261005
query_token_ablation_20261005 query_token_dynamic_20261005 query_token_dynamic_full_eval_20261007
query_token_staged_20261005 prefix_staged_20261008 prefix_policy_20261008
updater_validation_20261012 multiturn_validation_20261014'''.split()

NOTICE = (
    '> **無效結果註記（2026-10-16）**：本文件中使用「固定舊 base predictor＋scorer 預篩候選池」的'
    '實驗結果，以及依賴這些結果的方法比較、超參數選擇與成效結論，已依使用者指示撤回，'
    '不得用於 milestone 或後續方法結論。包含共用這些候選池的 base、Q0、memory、AdaJEPA、'
    'initial-scene 與 oracle 評估，以及 2026-10-16 的 deploy-pool 重跑。'
    '原始數據保留供追查；不代表數值遭捏造。獨立的訓練紀錄、工程正確性檢查與候選覆蓋率稽核'
    '不因此被判定錯誤。權威註記：`results/INVALID_RESULTS_20261016.json`。\n\n'
)


def main(root):
    results = root / 'results'
    record = {
        'status': 'INVALID', 'date': '2026-10-16', 'authority': 'Explicit user instruction',
        'reason': 'Historical fixed base predictor and scorer selected the candidate shortlist instead of the intended current-checkpoint deployment protocol.',
        'scope_rule': 'ALL experiment results and dependent claims using this historical-model shortlist protocol, including inherited cached shortlists. Applies across methods, folds, seeds and descendant artifacts.',
        'use_for_method_conclusions': False, 'use_for_milestone': False,
        'raw_artifacts_policy': 'Preserve raw metrics, immutable plans/code and checkpoint files; validity sidecars override their former reporting status.',
        'exceptions': ['Independent training records/checkpoint existence', 'Engineering equivalence and numerical checks', 'Official-cache and candidate-coverage audits documenting the problem'],
        'new_rerun': {'run': 'deploy_pool_updater_20261016', 'uses_rejected_selector': True,
                      'status': 'INVALID_CANCELLED', 'formal_training_started_before_cancellation': False},
        'replacement_protocol': 'Not launched. Do not silently substitute a new selector or resume cancelled runs.',
        'coverage_evidence': 'deploy_pool_updater_20261016/FIRST5000_SUCCESS_COVERAGE.json',
        'annotated_run_roots': [], 'annotated_legacy_results': [], 'annotated_documents': [],
    }
    for name in RUNS:
        folder = results / name
        if not folder.is_dir():
            continue
        sidecar = {k: v for k, v in record.items() if not k.startswith('annotated_')}
        sidecar.update(run=name, applies_to='All historical-shortlist experiment metrics and dependent conclusions in this directory and descendants; exceptions above remain explicit.')
        (folder / 'VALIDITY.json').write_text(json.dumps(sidecar, indent=2) + '\n')
        record['annotated_run_roots'].append(name)
    # Historical closed-loop reports also used base-selected candidate pools.
    for pattern in ('cx_clm_*.json', 'x3_clm_*.json', 'clm_*.json', 'w400_*.json'):
        for path in sorted(results.glob(pattern)):
            if path.name.endswith('.validity.json'):
                continue
            sidecar = {'status': 'INVALID', 'source': path.name,
                       'scope': 'Historical base-selected shortlist evaluation and dependent claims',
                       'authority': 'Explicit user instruction 2026-10-16',
                       'registry': 'INVALID_RESULTS_20261016.json', 'use_for_method_conclusions': False}
            path.with_suffix('.validity.json').write_text(json.dumps(sidecar, indent=2) + '\n')
            record['annotated_legacy_results'].append(path.name)
    docs = list((root / 'docs').glob('*.md'))
    docs += [p for name in RUNS for p in (results / name).glob('*.md')]
    docs += [root / 'README.md']
    tokens = ('shortlist', '1,024', '1024', 'matched_base_q0', 'milestone_memory', 'small_official',
              'AUCCESS', 'S@9', 'S@100')
    for path in docs:
        body = path.read_text()
        if path.name != 'README.md' and not any(t in body for t in tokens):
            continue
        if not body.startswith(NOTICE):
            path.write_text(NOTICE + body)
        record['annotated_documents'].append(str(path.relative_to(root)))
    (results / 'INVALID_RESULTS_20261016.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: len(v) for k, v in record.items() if k.startswith('annotated_')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
