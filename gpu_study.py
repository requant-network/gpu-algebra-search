"""Complete certifiable GPU random walks, paired policies and censored targets."""
import argparse,hashlib,json,pathlib,time
from datetime import datetime,timezone
import cupy as cp
import numpy as np
from experiments.tensor import kronecker_square,naive,verify

VARIANTS={
 'fixed1':dict(probes=1,generalized=0,adaptive=0,force_period=0,pool=False,cap=128),
 'fixed2':dict(probes=2,generalized=0,adaptive=0,force_period=0,pool=False,cap=128),
 'group1':dict(probes=1,generalized=1,adaptive=0,force_period=0,pool=False,cap=128),
 'group2':dict(probes=2,generalized=1,adaptive=0,force_period=0,pool=False,cap=128),
 'adaptive2':dict(probes=2,generalized=1,adaptive=1,force_period=0,pool=False,cap=128),
 'pool1':dict(probes=1,generalized=1,adaptive=0,force_period=0,pool=True,cap=128),
 'capacity96':dict(probes=1,generalized=1,adaptive=0,force_period=0,pool=False,cap=96)}

class State:
    def __init__(self,kernel,shape,seed,walks,cap,initial=None):
        self.kernel=kernel;self.shape=shape;self.walks=walks;self.cap=cap;self.offset=0
        terms=naive(shape) if initial is None else initial;assert len(terms)<cap and verify(terms,shape)
        initial=np.zeros((walks,cap,3),np.uint32);initial[:,:len(terms)]=np.array(terms,np.uint32)
        rng=np.random.default_rng(seed)
        self.current=cp.asarray(initial);self.best=self.current.copy()
        self.rng=cp.asarray(rng.integers(1,1<<32,(walks,32),np.uint32))
        self.ranks=cp.full(walks,len(terms),cp.int32);self.first=cp.full(walks,-1,cp.int32)
        self.ages=cp.zeros(walks,cp.int32);self.stats=cp.zeros((walks,6),cp.uint64)
        self.empty=cp.empty(1,cp.uint32);self.reseeds=0
    def advance(self,steps,config,target=0,trace=False):
        output=cp.empty((self.walks,steps,self.cap,3),cp.uint32) if trace else self.empty
        args=(self.current,self.best,self.rng,self.ranks,self.first,self.ages,self.stats,output,
              *[np.int32(x) for x in (steps,self.offset,self.walks,self.cap,config['probes'],config['generalized'],config['adaptive'],config['force_period'],target,trace)])
        self.kernel(((self.walks*32+127)//128,),(128,),args);self.offset+=steps
        return output
    def reseed_pool(self):
        ranks=cp.asnumpy(self.ranks);index=int(np.argmin(ranks));replace=cp.asarray(np.argsort(ranks)[-max(1,self.walks//4):])
        self.current[replace]=self.best[index];self.ages[replace]=0;self.reseeds+=1
    def certificates(self):
        best=cp.asnumpy(self.best);ranks=cp.asnumpy(self.ranks);schemes=[]
        for rows,rank in zip(best,ranks):
            scheme=[list(map(int,row)) for row in rows if row[0]]
            assert len(scheme)==int(rank) and verify(scheme,self.shape)
            schemes.append(scheme)
        return schemes,list(map(int,ranks))

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109);parser.add_argument('--trials',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=5);parser.add_argument('--walks',type=int,default=256)
    parser.add_argument('--variants',default=','.join(VARIANTS));parser.add_argument('--shapes',default='4,4,4;3,4,3')
    parser.add_argument('--start',choices=('naive','strassen49'),default='naive')
    parser.add_argument('--output',default='results/gpu-study.json');args=parser.parse_args()
    assert args.trials>0 and args.seconds>0 and args.walks>0
    module=cp.RawModule(path=str(pathlib.Path('build/search.cubin').resolve()));kernel=module.get_function('search')
    checks=0
    for name in ('fixed2','group2','adaptive2'):
        config=VARIANTS[name].copy()
        if name=='adaptive2':config['force_period']=16
        state=State(kernel,(4,4,4),args.seed,4,config['cap'])
        snapshots=cp.asnumpy(state.advance(200,config,trace=True))
        for walk in snapshots:
            for rows in walk:
                assert verify([list(map(int,row)) for row in rows if row[0]],(4,4,4)),name
                checks+=1
        state.certificates()
    cases=[]
    shapes=[tuple(map(int,part.split(','))) for part in args.shapes.split(';')]
    for shape in shapes:
      target=47 if shape==(4,4,4) else 32
      initial=None
      if args.start=='strassen49':
        assert shape==(4,4,4)
        base=json.loads(pathlib.Path('fixtures/alphatensor-2x2x2.json').read_text())['scheme']
        initial=kronecker_square(base,base,2,2);assert len(initial)==49 and verify(initial,shape)
      for trial in range(args.trials):
       names=args.variants.split(',')
       if trial%2:names.reverse()
       for name in names:
        config=VARIANTS[name];seed=args.seed+1009*trial
        state=State(kernel,shape,seed,args.walks,config['cap'],initial)
        # Warm the same kernel/config on a separate state, preserving trial initial conditions.
        warm=State(kernel,shape,seed+17,4,config['cap']);warm.advance(32,config);cp.cuda.get_current_stream().synchronize();del warm
        trajectory=[];device_ms=0;start=time.perf_counter();observed_hit=None;last_pool=0
        while time.perf_counter()-start<args.seconds:
            begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();state.advance(4096,config,target);end.record();end.synchronize()
            device_ms+=cp.cuda.get_elapsed_time(begin,end)
            ranks=cp.asnumpy(state.ranks);elapsed=(time.perf_counter()-start)*1000
            trajectory.append(dict(wall_ms=elapsed,best_rank=int(ranks.min()),target_walks=int((ranks<=target).sum()),step_bound=state.offset))
            if ranks.min()<=target:observed_hit=elapsed;break
            if config['pool'] and state.offset-last_pool>=65536:state.reseed_pool();last_pool=state.offset
        schemes,ranks=state.certificates();complete=(time.perf_counter()-start)*1000
        selected=int(np.argmin(ranks));best=schemes[selected];damaged=[row[:] for row in best];damaged[0][0]^=1;assert not verify(damaged,shape)
        cases.append(dict(shape=list(shape),start=args.start,initial_rank=len(naive(shape)) if initial is None else len(initial),trial=trial,seed=seed,variant=name,config=config,walks=args.walks,target_rank=target,
            target_hit=observed_hit is not None,observed_target_upper_bound_ms=observed_hit,censored_at_budget=observed_hit is None,
            nominal_budget_seconds=args.seconds,complete_verified_wall_ms=complete,device_elapsed_ms=device_ms,
            ranks=ranks,best_rank=min(ranks),best_scheme=best,every_walk_exact=True,pool_reseeds=state.reseeds,
            first_hit_steps=list(map(int,cp.asnumpy(state.first))),transition_counts=cp.asnumpy(state.stats).sum(axis=0).tolist(),trajectory=trajectory))
        print(shape,trial,name,'best',min(ranks),'hit',observed_hit,'verified wall',round(complete,2),flush=True)
    sources=['gpu_study.py','experiments/tensor.py','cuda/search.cu','fixtures/alphatensor-2x2x2.json']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},
        cubin_sha256=hashlib.sha256(pathlib.Path('build/search.cubin').read_bytes()).hexdigest(),transition_snapshot_checks=checks,cases=cases,
        scope='Original exact GF(2) flip/plus/shared-factor-reduction implementation, inspired by arXiv:2609.17533v2; not an execution of the paper authors GPU binary. Each trial is an independent ensemble; walks in pool policy share elite schemes. All policies paired by initial random seed and nominal wall budget. Target times are host-observed chunk upper bounds; unsuccessful trials remain censored. Complete verified wall includes final D2H and exact CPU verification of every walk, excludes initial allocation and warm-up. Lower rank alone is not a demonstrated hardware GEMM speedup.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
