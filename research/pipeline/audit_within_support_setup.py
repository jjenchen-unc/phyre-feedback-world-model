"""Independent exposure, shortlist and simulator alignment checks on actual smoke data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import phyre
import phyre.loader
from phyre import simulator as S
from phyre.interface.task.ttypes import Task
from thrift import TSerialization


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(t):return hashlib.sha256(S.serialize(t.scene)+repr((t.bodyId1,t.bodyId2,t.relationships)).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    sel=json.loads((root/'selection.json').read_text());assets=Path(sel['assets'])
    compiled=phyre.loader.load_compiled_task_dict();known={fingerprint(t) for t in compiled.values()};seen=set()
    for path,h in sel['exposure_sources'].items():
        assert sha(path)==h
        if path.endswith('.jsonl'):
            seen.update(json.loads(line)['task'] for line in Path(path).read_text().splitlines())
    fresh=set()
    for row in sel['rows']:
        path=root/'tasks'/(row['task'].replace(':','_')+'.bin');assert sha(path)==row['task_file_sha256']
        task=TSerialization.deserialize(Task(),path.read_bytes());assert task.taskId==row['task']
        assert task.taskId not in compiled and task.taskId not in seen
        h=fingerprint(task);assert h==row['scene_goal_sha256'] and h not in known and h not in fresh;fresh.add(h)
    checked=0;simulator_checks=0
    for path in sorted((assets/'smoke_scores').glob('*.npz')):
        task_id=path.stem.replace('_',':');task=TSerialization.deserialize(Task(),(root/'tasks'/(path.stem+'.bin')).read_bytes())
        sim=phyre.ActionSimulator([task],'ball')
        with np.load(path) as s, np.load(assets/'smoke_candidates'/path.name) as c:
            assert 'status' not in c and 'solved' not in c and c['frames'].ndim==3
            order=sorted(range(len(s['full_scores'])),key=lambda i:-s['full_scores'][i])[:8]
            np.testing.assert_array_equal(s['shortlist_positions'],order)
            np.testing.assert_array_equal(s['action_idx'],c['action_idx'][order])
            np.testing.assert_array_equal(s['action'],c['action'][order])
            for ai,act in zip(s['action_idx'],s['action']):
                j=int(np.flatnonzero(c['action_idx']==ai)[0]);file=assets/'smoke_data/rollouts'/f'{path.stem}_{int(ai):06d}.npz'
                with np.load(file) as r:
                    assert len(r['frames'])==128 and len(r['features'])==128
                    np.testing.assert_array_equal(r['frames'][0],c['frames'][j])
                    np.testing.assert_array_equal(r['features'][0],c['feats0'][j])
                    np.testing.assert_array_equal(r['action'],act)
                    status=int(sim.simulate_action(0,act,need_images=False).status)
                    assert status==int(r['status']) and status!=0;simulator_checks+=1
                checked+=1
    assert checked==16
    result={'passed':True,'selection_sha256':sha(root/'selection.json'),'fresh_scenes_checked':len(fresh),
        'candidate_to_rollout_pairs_checked':checked,'independent_simulator_labels_checked':simulator_checks,
        'checks':['no compiled or recorded task overlap','new serialized scene and goal','unique scene-goal fingerprints',
            'stable shortlist sorting without outcome fields','initial pixels and geometry equal recorded rollout',
            'actual simulator status replay'],'auditor_sha256':sha(__file__)}
    (root/'SETUP_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
