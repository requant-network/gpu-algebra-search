"""Convert attributed public AlphaTensor data to independently verified masks."""
import argparse,hashlib,json,pathlib,subprocess
import numpy as np
from experiments.tensor import verify

REVISION='1949163da3bef7e3eb268a3ac015fd1c2dbfc767'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('upstream',type=pathlib.Path);args=parser.parse_args()
assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
path=args.upstream/'algorithms/factorizations_f2.npz';archive=np.load(path,allow_pickle=False)
for key in ('2,2,2','4,4,4','3,3,3'):
    shape=list(map(int,key.split(',')));factors=archive[key];scheme=[]
    for rank in range(factors.shape[-1]):
        masks=[]
        for axis in range(3):
            vector=factors[axis,:,rank]
            if axis==2:vector=vector.reshape(shape[2],shape[0]).T.reshape(-1)
            masks.append(sum((int(value)&1)<<i for i,value in enumerate(vector)))
        scheme.append(masks)
    assert verify(scheme,shape),key
    result=dict(shape=shape,rank=len(scheme),scheme=scheme,origin='google-deepmind/alphatensor',revision=REVISION,
                data_file='algorithms/factorizations_f2.npz',data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                convention='Third factor transposed from trilinear trace orientation to row-major output.',license='Apache-2.0')
    pathlib.Path('fixtures/alphatensor-'+key.replace(',','x')+'.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
pathlib.Path('fixtures/ALPHATENSOR-LICENSE').write_bytes((args.upstream/'LICENSE').read_bytes())
