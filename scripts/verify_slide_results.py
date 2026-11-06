"""Audit documented slide tables against published per-scene results.

Standard library only. Optional --deck checks the original local PPTX too.
This verifies recorded results, not simulator or raw-score replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
METRICS = (1, 3, 5, 9)
DECK_SHA256 = '8336f484dcf5c8195c9eb3c286d1098b87f186af745e08e4a0993d87d0d1a144'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def recount(method):
    ranks = method['first_success']
    assert len(ranks) == method['n'] == 100
    counts = [sum(rank is not None and rank <= k for rank in ranks.values())
              for k in METRICS]
    assert counts == [method['counts'][f's{k}'] for k in METRICS]
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', type=Path)
    args = parser.parse_args()
    base = read('research/results/plain_base_comparison_20261019/SUMMARY.json')
    loss = read('research/results/loss_design_iteration_20261019/FINAL_SUMMARY.json')
    milestone = read('results/milestone.json')
    assert base['complete'] and loss['complete']
    specifications = [(base, 'base'), (base, 'q0'), (base, 'tta'),
                      (loss, 'l1'), (loss, 'l1_centered'),
                      (loss, 'smooth_centered'), (loss, 'smooth'), (loss, 'smooth_zero')]
    rows = [recount(data['panels']['within']['methods'][method]) +
            recount(data['panels']['cross']['methods'][method])
            for data, method in specifications]
    doc = (ROOT / 'docs/EXPERIMENT_FLOW_AND_SLIDE_ALIGNMENT.md').read_text()
    documented = []
    for line in doc.splitlines():
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) == 9 and all(cell.isdigit() for cell in cells[1:]):
            documented.append([int(cell) for cell in cells[1:]])
    assert rows == documented, 'Documentation table differs from per-scene counts'

    for panel in ['within', 'cross']:
        assert base['panels'][panel]['methods']['uniform']['first_success'] == \
            loss['panels'][panel]['methods']['l1']['first_success']
        assert base['panels'][panel]['methods']['uniform_centered']['first_success'] == \
            loss['panels'][panel]['methods']['l1_centered']['first_success']
    seed1 = loss['panels']['within']['seed1']
    assert [recount(seed1[m])[-1] for m in ['l1', 'smooth']] == [58, 50]
    assert not loss['panels']['cross']['seed1']

    oracle = []
    for panel in ['within', 'cross']:
        data = milestone['oracle_1024'][panel]
        assert len(data['first_success']) == data['n']
        for method in ['q0', 'oracle_latent']:
            rate = sum(r[method] is not None and r[method] <= 9
                       for r in data['first_success'].values()) / data['n']
            assert abs(rate - data['methods'][method]['S9']) < 1e-12
            oracle.append(round(100 * rate))
    assert oracle == [60, 96, 20, 100]
    coverage = []
    for panel in ['cross', 'within']:
        for method in ['base', 'q0', 'scene']:
            data = milestone['coverage_100k'][panel][method]
            assert data['n_tasks'] == 100
            assert abs(data['covered_tasks']/100 - data['success_at']['1024']) < 1e-12
            coverage.append(data['covered_tasks'])
    assert coverage == [54, 51, 61, 94, 91, 98]

    if args.deck:
        assert hashlib.sha256(args.deck.read_bytes()).hexdigest() == DECK_SHA256
        expected = {7: [n for row in rows[:6] for n in [row[3], row[7]]],
                    8: oracle, 11: [n for row in rows for n in row], 13: coverage}
        with ZipFile(args.deck) as deck:
            for slide, values in expected.items():
                root = ElementTree.fromstring(deck.read(f'ppt/slides/slide{slide}.xml'))
                texts = [e.text or '' for e in root.iter(
                    '{http://schemas.openxmlformats.org/drawingml/2006/main}t')]
                # Table cells only: explanatory sentences may contain other percentages.
                actual = [int(s[:-1]) for s in texts if re.fullmatch(r'\d+%', s)]
                assert actual == values, (slide, actual, values)
        print('Original PPTX hash and slides 7/8/11/13 table cells agree.')
    print('64 documented result cells, second-seed counts, oracle ranks and coverage agree.')
    print('Coverage checked against published aggregates; no raw-label replay performed.')


if __name__ == '__main__':
    main()
