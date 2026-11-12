"""Complete batched GF2 matrix products from a rank-47 certificate."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cupy as cp
import numpy as np
from experiments.tensor import verify


def reference(a,b):
    result=np.zeros_like(a)
    for i in range(4):
        for j in range(4):
            bit=np.zeros_like(a)
            for k in range(4):bit^=((a>>(4*i+k))&(b>>(4*k+j)))&1
            result|=bit<<(4*i+j)
    return result


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--output',default='results/compiled-study.json');args=parser.parse_args()
    fixture=Path('fixtures/discovered-rank47.json');scheme=json.loads(fixture.read_text())['scheme']
    assert len(scheme)==47 and verify(scheme,(4,4,4))
    build_path=Path('results/compiled-build.json');build=json.loads(build_path.read_text())
    for p,digest in build['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
    binary=Path('build/compiled_scheme.cubin');assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['cubin_sha256']
    module=cp.RawModule(path=str(binary.resolve()));kernels={name:module.get_function(name) for name in ('direct','factorized')}
    # All A patterns against each basis B directly exercise the full tensor identity on GPU.
    a=np.tile(np.arange(65536,dtype=np.uint16),16);b=np.repeat(np.array([1<<i for i in range(16)],np.uint16),65536)
    da,db=cp.asarray(a),cp.asarray(b);out=cp.empty_like(da);expected=reference(a,b)
    for kernel in kernels.values():
        kernel(((len(a)+255)//256,),(256,),(da,db,out,np.int32(len(a))))
        assert np.array_equal(cp.asnumpy(out),expected)
    validation_pairs=len(a);rng=np.random.default_rng(args.seed);cases=[]
    for n in (4096,262144,1048576):
        a=rng.integers(0,65536,n,np.uint16);b=rng.integers(0,65536,n,np.uint16)
        a[:4]=[0,65535,33825,1];b[:4]=[65535,65535,33825,1]
        expected=reference(a,b);da,db=cp.asarray(a),cp.asarray(b);out=cp.empty_like(da)
        samples={name:{'resident_device_ms':[],'offload_wall_ms':[]} for name in kernels}
        for iteration in range(35):
            names=list(kernels)
            if iteration%2:names.reverse()
            for name in names:
                kernel=kernels[name];start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
                kernel(((n+255)//256,),(256,),(da,db,out,np.int32(n)));end.record();end.synchronize()
                device=cp.cuda.get_elapsed_time(start,end);assert np.array_equal(cp.asnumpy(out),expected)
                begin=time.perf_counter();da.set(a);db.set(b)
                kernel(((n+255)//256,),(256,),(da,db,out,np.int32(n)));actual=out.get()
                full=(time.perf_counter()-begin)*1000;assert np.array_equal(actual,expected)
                if iteration>=5:samples[name]['resident_device_ms'].append(device);samples[name]['offload_wall_ms'].append(full)
        cases.append(dict(n=n,input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(),
                          output_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),every_output_exact=True,samples=samples))
        print(n,{name:float(np.median(scopes['resident_device_ms'])) for name,scopes in samples.items()},flush=True)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
                exhaustive_a_basis_b_pairs=validation_pairs,
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ('compiled_study.py','compile_scheme.py','experiments/tensor.py','fixtures/discovered-rank47.json')},
                result_sha256={build_path.name:hashlib.sha256(build_path.read_bytes()).hexdigest()},
                scope='Independent batched4x4 scalar GF2 products. Discovered rank47 tensor compiled to specialized mask/parity operations; comparator uses direct row-column popcount. Complete scalar product, not a blocked large-matrix GEMM or FP8 arithmetic. Resident is one device kernel; offload adds both input uploads and output download with persistent allocations. Setup and CPU validation excluded.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
