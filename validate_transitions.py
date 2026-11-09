"""Match actual GPU state transitions to independent CPU transformations."""
import collections,hashlib,itertools,json,pathlib
import cupy as cp
import numpy as np
from gpu_study import State,VARIANTS
from experiments.tensor import flip,naive,plus,reduce_group,verify

def bag(scheme):return collections.Counter(map(tuple,scheme))

def match(before,after):
    expected=bag(after);previous=bag(before)
    if previous==expected:return 'unchanged'
    removed=previous-expected
    remaining=removed.copy();indices=[]
    for i,row in enumerate(before):
        if remaining[tuple(row)]>0:indices.append(i);remaining[tuple(row)]-=1
    if sum(removed.values())<=2:
        for i,j in itertools.permutations(indices,2):
            for axis in range(3):
                if before[i][axis]==before[j][axis] and bag(flip(before,i,j,axis))==expected:return 'flip'
    if len(after)==len(before)+1:
        for j in indices:
            for i in range(len(before)):
                if i!=j:
                    for axis in range(3):
                        if bag(plus(before,i,j,axis))==expected:return 'plus'
    if len(after)<len(before):
        for axis in range(3):
            groups=collections.defaultdict(list)
            for i,row in enumerate(before):groups[row[axis]].append(i)
            candidates=[]
            if 2<=len(indices)<=5 and len({before[i][axis] for i in indices})==1:candidates.append(indices)
            for group in groups.values():
                if len(group)<=8:
                    for size in range(2,min(5,len(group))+1):candidates.extend(itertools.combinations(group,size))
            for selected in candidates:
                value=0
                for i in selected:value^=before[i][(axis+1)%3]
                if value:continue
                for pivot in selected:
                    order=[pivot]+[i for i in selected if i!=pivot]
                    if bag(reduce_group(before,order,axis,(1<<len(order))-1))==expected:return 'generalized_reduction'
    raise AssertionError(('No independent CPU transition matches',removed,expected-previous))

assert __debug__
module=cp.RawModule(path=str(pathlib.Path('build/search.cubin').resolve()));kernel=module.get_function('search')
cases=[]
for kind in ('flip','plus','generalized'):
    config=VARIANTS['group1'].copy();initial=naive((4,4,4))
    if kind=='flip':config['generalized']=0
    if kind=='plus':config['force_period']=1
    if kind=='generalized':
        for i in range(7):initial.extend([[0xff00|i,1,5],[0xff00|i,2,5],[0xff00|i,3,5]])
    assert verify(initial,(4,4,4))
    state=State(kernel,(4,4,4),20261109,8,128,initial)
    if kind=='plus':state.ages.fill(1000)
    snapshots=cp.asnumpy(state.advance(64,config,trace=True));matches=collections.Counter()
    for walk in snapshots:
        before=initial
        for rows in walk:
            after=[list(map(int,row)) for row in rows if row[0]];assert verify(after,(4,4,4))
            matches[match(before,after)]+=1;before=after
    state.certificates();counts=cp.asnumpy(state.stats).sum(axis=0).tolist()
    cases.append(dict(kind=kind,cpu_matched_transitions=dict(matches),gpu_branch_counts=counts))
    print(kind,dict(matches),counts,flush=True)
assert cases[1]['cpu_matched_transitions'].get('plus',0)>0
assert cases[2]['gpu_branch_counts'][2]>0 and cases[2]['cpu_matched_transitions'].get('generalized_reduction',0)>0
sources=['validate_transitions.py','gpu_study.py','experiments/tensor.py','cuda/search.cu']
record=dict(cases=cases,total_matched_transitions=8*64*3,source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},
    scope='Actual production GPU kernel snapshots matched against independent CPU flip, plus and shared-factor reduction. Extra zero-sum triads deliberately exercise group reduction; forced-plus fixture exercises insertion. Exact tensor checked after every transition, including slot shuffles.')
pathlib.Path('results/transition-validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
