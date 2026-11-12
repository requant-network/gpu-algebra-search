"""Exploratory paired success-count analysis; no fixed-input latency inference."""
import csv
import hashlib
import json
import math
from pathlib import Path

def paired_p(control_only,candidate_only):
    assert control_only>=0 and candidate_only>=0
    n=control_only+candidate_only
    return min(1.,2*sum(math.comb(n,k) for k in range(min(control_only,candidate_only)+1))/2**n) if n else 1.

def main():
    assert paired_p(0,0)==paired_p(1,0)==1 and paired_p(2,0)==.5 and paired_p(4,0)==.125
    paths=[Path('results')/f'diversity-{suffix}.json' for suffix in ('study','replication')];pairs=[];seen_seeds=set()
    for path in paths:
        record=json.loads(path.read_text());trials={}
        for c in record['cases']:trials.setdefault(c['trial'],{})[c['variant']]=c
        assert len(trials)==32
        for trial in trials.values():
            assert set(trial)=={'fixed1','group1','adaptive2','pool1','diverse8'}
            assert len({c['seed'] for c in trial.values()})==1
            seed=next(iter(trial.values()))['seed'];assert seed not in seen_seeds;seen_seeds.add(seed)
            pairs.append({name:c['target_hit'] for name,c in trial.items()})
    comparisons=[]
    for control in ('fixed1','group1','adaptive2','pool1'):
        b=sum(p[control] and not p['diverse8'] for p in pairs);c=sum(p['diverse8'] and not p[control] for p in pairs)
        comparisons.append(dict(control=control,paired_trials=len(pairs),control_only=b,candidate_only=c,
                                both_success=sum(p[control] and p['diverse8'] for p in pairs),
                                neither_success=sum(not p[control] and not p['diverse8'] for p in pairs),two_sided_exact_p=paired_p(b,c)))
    previous=0
    for index,row in enumerate(sorted(comparisons,key=lambda r:r['two_sided_exact_p'])):
        previous=max(previous,min(1.,(len(comparisons)-index)*row['two_sided_exact_p']));row['holm_adjusted_p']=previous
    record=dict(candidate='diverse8',comparisons=comparisons,
                source_sha256={'policy_statistics.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                result_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                scope='Exploratory post-hoc exact two-sided McNemar/binomial test on64 paired seeded trials, with Holm adjustment over the four candidate-versus-control comparisons. Target hit at the declared5-second budget;failed/censored trials retained. No pre-registration,universal policy ranking,asymptotic search or fixed-input latency confidence claim.')
    Path('results/policy-statistics.json').write_text(json.dumps(record,indent=2)+'\n')
    with Path('results/policy-summary.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(comparisons[0]),lineterminator='\n');writer.writeheader();writer.writerows(comparisons)
    print('64 paired trials; exploratory four-comparison exact/Holm table generated')

if __name__=='__main__':main()
