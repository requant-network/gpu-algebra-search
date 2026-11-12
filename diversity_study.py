"""Larger paired cohorts, adding a population that retains distinct elites."""
import hashlib
import json
from pathlib import Path
import sys
import cupy as cp
import numpy as np
import gpu_study as study
from experiments.diversity import elites


class State(study.State):
    def advance(self, steps, config, target=0, trace=False):
        self.diverse=config.get('diverse',False)
        return super().advance(steps,config,target,trace)

    def reseed_pool(self):
        if not self.diverse:
            return super().reseed_pool()
        ranks=cp.asnumpy(self.ranks)
        selected=elites(cp.asnumpy(self.best),ranks,8)
        replace=np.argsort(ranks)[-max(1,self.walks//4):]
        source=np.resize(np.array(selected,np.int32),len(replace))
        target=cp.asarray(replace)
        self.current[target]=self.best[cp.asarray(source)]
        self.ages[target]=0
        self.reseeds+=1


if __name__=='__main__':
    assert __debug__ and '--output' in sys.argv, 'An explicit fresh output is required'
    output=Path(sys.argv[sys.argv.index('--output')+1]);assert not output.exists(), 'Keep prior campaign evidence'
    study.State=State
    study.VARIANTS['diverse8']={**study.VARIANTS['pool1'],'diverse':True,'elite_limit':8}
    study.main()
    result=json.loads(output.read_text())
    result['source_sha256'].update({p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in ('diversity_study.py','experiments/diversity.py')})
    result['scope']+=' Diverse8 canonicalizes term order, retains up to eight distinct best schemes and reseeds the worst quarter cyclically from them; kernel transitions and initial budgets stay unchanged.'
    output.write_text(json.dumps(result,indent=2)+'\n')
