"""Independently recheck every retained GPU best-scheme certificate on CPU."""
import json,pathlib
from experiments.tensor import verify
count=0;hits=0
for path in sorted(pathlib.Path('results').glob('gpu-*.json')):
    record=json.loads(path.read_text(encoding='utf-8'))
    for case in record['cases']:
        scheme=case['best_scheme'];shape=case['shape']
        assert len(scheme)==case['best_rank']==min(case['ranks'])
        assert verify(scheme,shape),(path,case['trial'],case['variant'])
        damaged=[row[:] for row in scheme];damaged[0][0]^=1;assert not verify(damaged,shape)
        assert case['target_hit']==(case['best_rank']<=case['target_rank'])
        assert case['censored_at_budget']!=case['target_hit']
        assert case['complete_verified_wall_ms']>=case['trajectory'][-1]['wall_ms']
        assert all(a['wall_ms']<=b['wall_ms'] and a['step_bound']<b['step_bound'] for a,b in zip(case['trajectory'],case['trajectory'][1:]))
        if case['target_hit']:assert case['observed_target_upper_bound_ms']==case['trajectory'][-1]['wall_ms']
        count+=1;hits+=case['target_hit']
assert count
print(count,'retained exact GPU certificates verified;',hits,'target successes; altered certificates rejected')
