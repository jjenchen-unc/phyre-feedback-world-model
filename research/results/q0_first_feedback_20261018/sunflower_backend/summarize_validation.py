"""Require all real backend and updater checks to match the current code."""
from pathlib import Path
import json,hashlib,statistics,os
B=Path(__file__).resolve().parent;R=B.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
policy=read(B/'ENCODING_POLICY.json')
assert policy['runtime']==sha(B/'runtime.py') and policy['rpc']==sha(B/'encoder_rpc.py')
assert policy['plan']==sha(R/'PLAN.json')
rpc=read(B/'RPC_VALIDATION.json')
assert rpc['passed'] and rpc['encoder_rpc']==policy['rpc'] and rpc['plan']==policy['plan']
assert rpc['four_concurrent_clients_exact']
checks=[];provenance=None
for panel in ['cross','within']:
    hw=read(B/f'READER_CHECK_{panel}.json')
    assert hw['passed'] and hw['max_abs']==0
    for kind in ['residual','zero']:
        path=R/'preflight'/panel/kind/'complete.json';x=read(path)
        prov=x['provenance']
        assert prov['execution_backend']==policy['runtime'] and prov['encoder_rpc']==policy['rpc']
        assert prov['encoding_policy']==sha(B/'ENCODING_POLICY.json') and prov['plan']==policy['plan']
        assert hw['provenance']==prov
        for key in ['passed','q0_first','training_eval_initial_scores_exact','query_disjoint','current_residual_replay_exact',
                    'checkpoint_gradient_exact','stress8_real_failures','reader_probe_head_frozen','tta_update_and_reset_verified']:
            assert x[key],(panel,kind,key)
        if provenance is not None:assert provenance==prov
        provenance=prov
        checks.append({'panel':panel,'kind':kind,'receipt_sha256':sha(path),'seconds':x['seconds'],
                       'peak_GiB':x['peak_GiB'],'stress_task':x['stress_task'],
                       'hot_cache_pilot_seconds_per_slot':[v['seconds'] for v in x['natural_cost_pilot']]})
out={'passed':True,'provenance':provenance,'rpc_reference_videos':len(rpc['records']),
     'exact_full_and_initial_latents':all(v['full_latent_exact'] and v['initial_exact'] for v in rpc['records']),
     'four_concurrent_clients_exact':True,'reader_score_max_abs':0.,'all_four_training_preflights':checks,
     'encoder_seconds_median':statistics.median(v['encode_s'] for v in rpc['records']),
     'transport_and_codec_seconds_median':rpc['median_transport_and_codec_s'],
     'transport_and_codec_seconds_max':rpc['max_transport_and_codec_s'],
     'training_scope':'Predictor/updater on A5000, canonical observation encoder on A6000; same registered sampling/loss/models',
     'limits':'Six parity videos and four engineering preflights; not model effectiveness evidence. Two-slot timing uses hot caches and is not a full-training ETA.'}
tmp=B/f'VALIDATION_SUMMARY.{os.getpid()}.tmp';tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(B/'VALIDATION_SUMMARY.json')
print(json.dumps(out,indent=2))
