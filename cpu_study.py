"""Exact certificate checks, independent matrix products and transition invariants."""
import argparse,hashlib,json,pathlib,random
from experiments.tensor import evaluate,flip,multiply,naive,plus,reduce_group,tensor,verify

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='results/cpu-study.json');args=parser.parse_args()
    rng=random.Random(20261109);certificates=[];checks=0
    for path in sorted(pathlib.Path('fixtures').glob('*.json')):
        record=json.loads(path.read_text());shape=record['shape'];scheme=record['scheme'];assert verify(scheme,shape)
        broken=[row[:] for row in scheme];broken[0][0]^=1
        assert not verify(broken,shape)
        for _ in range(100):
            a=rng.getrandbits(shape[0]*shape[1]);b=rng.getrandbits(shape[1]*shape[2])
            assert evaluate(scheme,a,b)==multiply(a,b,shape);checks+=1
        certificates.append(dict(file=path.name,rank=len(scheme),shape=shape,exact_tensor=True))
    assert certificates,'Published certificates are required'
    transitions=0
    for shape in ((2,2,2),(4,4,4),(3,4,3)):
        scheme=naive(shape);expected=tensor(scheme,shape)
        for step in range(300):
            i,j=rng.sample(range(len(scheme)),2);axis=rng.randrange(3)
            matches=[t for t in range(len(scheme)) if t!=i and scheme[t][axis]==scheme[i][axis]]
            scheme=flip(scheme,i,rng.choice(matches),axis) if matches and rng.random()<.9 else plus(scheme,i,j,axis)
            assert tensor(scheme,shape)==expected;transitions+=1
    group=[[1,1,1],[1,2,2],[1,3,4]];before=tensor(group,(2,2,2))
    assert tensor(reduce_group(group,[0,1,2],0,7),(2,2,2))==before
    for a in range(16):
        for b in range(16):assert evaluate(naive((2,2,2)),a,b)==multiply(a,b,(2,2,2));checks+=1
    sources=['cpu_study.py','experiments/tensor.py']+[str(p).replace('\\','/') for p in sorted(pathlib.Path('fixtures').glob('*.json'))]
    record=dict(certificates=certificates,matrix_product_checks=checks,exact_transition_checks=transitions,
        source_sha256={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in sources},
        scope='Exact tensor identity and independent GF(2) matrix evaluation. Published schemes are attributed references, not discoveries by this project.')
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(len(certificates),'exact certificates;',checks,'matrix products;',transitions,'invariant-preserving transitions passed')

if __name__=='__main__':main()
