"""CPU replay of follow-up evidence, provenance and derived timing summaries."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
assert __debug__
PATTERNS = ('compiled-*.json','diversity-*.json','wide-*.json')
paths=sorted(set(p for pattern in PATTERNS for p in Path('results').glob(pattern)))
assert len(paths)==7,(len(paths),[p.name for p in paths])
records={p.name:json.loads(p.read_text()) for p in paths}
def binding(name,expected,result=False):
 p=(Path('results')/name) if result else Path(name)
 assert not Path(name).is_absolute() and ':' not in name and '..' not in Path(name).parts
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,(p,expected)
def provenance(r):
 if isinstance(r,dict):
  for key,v in r.items():
   if key in ('source_sha256','artifacts_sha256','public_artifacts_sha256','result_sha256'):
    for name,expected in v.items():binding(name,expected,key=='result_sha256')
   elif isinstance(v,(dict,list)):provenance(v)
 elif isinstance(r,list):
  for v in r:provenance(v)
for r in records.values():provenance(r)
from experiments.tensor import verify
fixture=json.loads(Path('fixtures/discovered-rank47.json').read_text());assert len(fixture['scheme'])==47 and verify(fixture['scheme'],fixture['shape'])
for name,r in records.items():
 if name.startswith('compiled-') and 'cases' in r:
  assert r['exhaustive_a_basis_b_pairs']==1048576 and len(r['cases'])==3
  assert all(c['every_output_exact'] for c in r['cases'])
 if name.startswith(('diversity-','wide-')):
  assert len(r['cases'])==(160 if name.startswith('diversity') else 48)
  if name.startswith('wide'):assert r['transition_32_64_equal_snapshots']==800 and r['wide_tensor_snapshots_checked']==400
  for c in r['cases']:
   assert c['every_walk_exact'] and len(c['best_scheme'])==c['best_rank']==min(c['ranks'])
   assert verify(c['best_scheme'],c['shape'])
   broken=[term[:] for term in c['best_scheme']];broken[0][0]^=1;assert not verify(broken,c['shape'])
   assert c['target_hit']==(c['best_rank']<=c['target_rank']) and c['censored_at_budget']!=c['target_hit']

rows=[]
def timings(value,labels,route=''):
 if isinstance(value,list):
  assert len(value)==30 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in value),(labels,route)
  rows.append(dict(**labels,variant_scope=route,median_ms=median(value),p95_ms=sorted(value)[28],repetitions=len(value)))
 elif isinstance(value,dict):
  for key,v in value.items():timings(v,labels,(route+'/' if route else '')+key)
for name,r in records.items():
 for index,c in enumerate(r.get('cases',[])):
  labels=dict(file=name,case=index,seed=c.get('seed',r.get('seed','')),**{key:c.get(key,'') for key in ('application','n','m','k','p','columns','steps','payload_bytes','batch','variant','pattern')})
  v=c.get('result',c)
  for key in ('samples','samples_ms','resident_graph_device_ms'):
   if isinstance(v.get(key),(dict,list)):timings(v[key],labels,key)
if rows:
 with Path('results/followup-summary.csv').open('w',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(records),'follow-up records validated;',len(rows),'median/p95 timing rows')
