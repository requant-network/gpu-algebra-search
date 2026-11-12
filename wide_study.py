"""Typed 64-bit factor extension, preserving 32-bit lane masks and RNG."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import cupy as cp
import numpy as np
import gpu_study as original
from experiments.tensor import naive,verify

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--output',default='results/wide-study.json');parser.add_argument('--nvcc',default='nvcc');args=parser.parse_args()
    source=Path('cuda/search.cu').read_text()
    replacements={
        'unsigned* schemes,unsigned* best_schemes':'unsigned long long* schemes,unsigned long long* best_schemes',
        'unsigned* trace':'unsigned long long* trace',
        'unsigned factors[4][3]':'unsigned long long factors[16][3]',
        's<4;s++':'s<16;s++',
        'unsigned value=slot':'unsigned long long value=slot',
        'unsigned values[5]':'unsigned long long values[5]',
        'unsigned sum=0':'unsigned long long sum=0',
        'unsigned local=slot':'unsigned long long local=slot',
        'unsigned pivot_value=':'unsigned long long pivot_value=',
        'unsigned av=':'unsigned long long av=',
        ';unsigned bw=':';unsigned long long bw=',
        'unsigned replacement[3]':'unsigned long long replacement[3]',
        'unsigned pivot=':'unsigned long long pivot='}
    for old,new in replacements.items():
        assert source.count(old)==(2 if old=='unsigned local=slot' else 1),(old,source.count(old))
        source=source.replace(old,new)
    path=Path('build/search64.cu');path.write_text(source);binary=path.with_suffix('.cubin')
    subprocess.run([args.nvcc,'-O3','-std=c++17','-arch=sm_120','-cubin',str(path),'-o',str(binary)],check=True)
    kernel=cp.RawModule(path=str(binary.resolve())).get_function('search')
    state_source=inspect.getsource(original.State)
    for old,new in (
        ('initial=np.zeros((walks,cap,3),np.uint32)','initial=np.zeros((walks,cap,3),np.uint64)'),
        ('np.array(terms,np.uint32)','np.array(terms,np.uint64)'),
        ('self.empty=cp.empty(1,cp.uint32)','self.empty=cp.empty(1,cp.uint64)'),
        ('cp.empty((self.walks,steps,self.cap,3),cp.uint32)','cp.empty((self.walks,steps,self.cap,3),cp.uint64)')):
        assert state_source.count(old)==1;state_source=state_source.replace(old,new)
    namespace=dict(vars(original));exec(compile(state_source,'<typed-64bit-state>','exec'),namespace);State=namespace['State']
    config=original.VARIANTS['group1'];small32=original.State(cp.RawModule(path=str(Path('build/search.cubin').resolve())).get_function('search'),(4,4,4),args.seed,4,128)
    small64=State(kernel,(4,4,4),args.seed,4,128)
    a=cp.asnumpy(small32.advance(200,config,trace=True));b=cp.asnumpy(small64.advance(200,config,trace=True))
    assert np.array_equal(a,b),'32/64 transition disagreement'
    checks=0
    for shape,cap in (((6,6,6),256),((8,4,8),288)):
        probe=State(kernel,shape,args.seed,2,cap);snapshots=cp.asnumpy(probe.advance(100,config,trace=True))
        for walk in snapshots:
            for rows in walk:assert verify([list(map(int,row)) for row in rows if row[0]],shape);checks+=1
        probe.certificates()
    cases=[]
    for shape,cap in (((6,6,6),256),((8,4,8),288)):
        for trial in range(8):
            names=['fixed1','group1','adaptive2']
            if trial%2:names.reverse()
            for name in names:
                config=original.VARIANTS[name];seed=args.seed+1009*trial;state=State(kernel,shape,seed,128,cap)
                warm=State(kernel,shape,seed+17,2,cap);warm.advance(32,config);cp.cuda.get_current_stream().synchronize();del warm
                import time
                start=time.perf_counter();trajectory=[];target=len(naive(shape))-1;hit=None
                while time.perf_counter()-start<2:
                    state.advance(2048,config,target);cp.cuda.get_current_stream().synchronize()
                    rank=int(state.ranks.min().get());elapsed=(time.perf_counter()-start)*1000
                    trajectory.append(dict(wall_ms=elapsed,best_rank=rank,step_bound=state.offset))
                    if rank<=target:hit=elapsed;break
                schemes,ranks=state.certificates();best=int(np.argmin(ranks))
                cases.append(dict(shape=shape,cap=cap,walks=128,variant=name,trial=trial,seed=seed,target_rank=target,
                                  nominal_budget_seconds=2,best_rank=ranks[best],best_scheme=schemes[best],ranks=ranks,
                                  every_walk_exact=True,target_hit=hit is not None,censored_at_budget=hit is None,
                                  observed_target_upper_bound_ms=hit,complete_verified_wall_ms=(time.perf_counter()-start)*1000,trajectory=trajectory))
                record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,cases=cases,transition_32_64_equal_snapshots=800,
                            wide_tensor_snapshots_checked=checks,generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                            transformed_state_sha256=hashlib.sha256(state_source.encode()).hexdigest(),cubin_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                            source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('wide_study.py','gpu_study.py','cuda/search.cu','experiments/tensor.py')},
                            scope='64-bit factors only: warp ballots, RNG and lane masks remain32-bit. Up to16 factor slots/lane. Eight seeded trials per policy/shape; target is any improvement over the naive rank, not a record claim. Exact full tensor checks include bit63; all128 final walk certificates independently checked. 32/64 transition identity on4x4 serves as regression control.')
                Path(args.output).write_text(json.dumps(record,indent=2)+'\n')
                print(shape,trial,name,'best',ranks[best],'verified',flush=True)

if __name__=='__main__':main()
