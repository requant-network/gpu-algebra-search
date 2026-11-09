"""Derive policy outcomes without dropping censored search trials."""
import collections,csv,json,math,pathlib,statistics

groups=collections.defaultdict(list)
for name in ('gpu-study.json','gpu-replication.json'):
    record=json.loads((pathlib.Path('results')/name).read_text(encoding='utf-8'))
    for case in record['cases']:groups[(record['seed'],tuple(case['shape']),case['start'],case['variant'])].append(case)
rows=[]
for (seed,shape,start,variant),cases in sorted(groups.items()):
    n=len(cases);success=sum(c['target_hit'] for c in cases);fraction=success/n;z=1.959963984540054
    denominator=1+z*z/n;center=(fraction+z*z/(2*n))/denominator
    radius=z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/denominator
    hits=[c['observed_target_upper_bound_ms'] for c in cases if c['target_hit']]
    restricted=[min(c['observed_target_upper_bound_ms'],c['nominal_budget_seconds']*1000) if c['target_hit'] else c['nominal_budget_seconds']*1000 for c in cases]
    rows.append(dict(seed=seed,shape='x'.join(map(str,shape)),start=start,variant=variant,trials=n,successes=success,
        success_fraction=fraction,wilson95_low=max(0,center-radius),wilson95_high=min(1,center+radius),
        median_final_rank=statistics.median(c['best_rank'] for c in cases),
        restricted_target_mean_ms=statistics.mean(restricted),
        median_successful_observation_ms=statistics.median(hits) if hits else '',
        median_complete_verified_wall_ms=statistics.median(c['complete_verified_wall_ms'] for c in cases)))
with pathlib.Path('results/summary.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(rows),'policy/cohort summaries; failures retained as censored trials')
