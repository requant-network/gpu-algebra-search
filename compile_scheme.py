"""Compile one independently certified discovered rank-47 scheme into CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from experiments.tensor import verify


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvcc', default='nvcc')
    args = parser.parse_args()
    record = json.loads(Path('results/gpu-study.json').read_text())
    candidates = [(index, case) for index, case in enumerate(record['cases'])
                  if case['shape']==[4,4,4] and case['best_rank']==47]
    assert candidates
    index, case = min(candidates, key=lambda pair: sorted(pair[1]['best_scheme']))
    scheme = sorted(case['best_scheme']); assert len(scheme)==47 and verify(scheme,(4,4,4))
    broken = [row[:] for row in scheme]; broken[0][0] ^= 1; assert not verify(broken,(4,4,4))
    fixture = dict(shape=[4,4,4],scheme=scheme,source_record='results/gpu-study.json',case_index=index,
                   source_record_sha256=hashlib.sha256(Path('results/gpu-study.json').read_bytes()).hexdigest())
    Path('fixtures/discovered-rank47.json').write_text(json.dumps(fixture,indent=2)+'\n')
    source = '''#include <cuda_runtime.h>
extern "C" __global__ void direct(const unsigned short* a,const unsigned short* b,unsigned short* out,int n){
 int t=blockIdx.x*256+threadIdx.x;if(t>=n)return;unsigned x=a[t],y=b[t],z=0;
 #pragma unroll
 for(int j=0;j<4;j++){
  unsigned col=((y>>j)&1)|(((y>>(4+j))&1)<<1)|(((y>>(8+j))&1)<<2)|(((y>>(12+j))&1)<<3);
  #pragma unroll
  for(int i=0;i<4;i++)z|=(__popc(((x>>(4*i))&15)&col)&1)<<(4*i+j);
 }out[t]=z;
}
extern "C" __global__ void factorized(const unsigned short* a,const unsigned short* b,unsigned short* out,int n){
 int t=blockIdx.x*256+threadIdx.x;if(t>=n)return;unsigned x=a[t],y=b[t],z=0;
'''
    for a,b,c in scheme:
        source += f' z^=(0U-((__popc(x&{a}U)&__popc(y&{b}U))&1U))&{c}U;\n'
    source += ' out[t]=z;\n}\n'
    Path('build').mkdir(exist_ok=True); generated=Path('build/compiled_scheme.cu'); generated.write_text(source)
    subprocess.run([args.nvcc,'-O3','-std=c++17','-arch=sm_120','-cubin',str(generated),'-o','build/compiled_scheme.cubin'],check=True)
    build = dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                 ('compile_scheme.py','experiments/tensor.py','fixtures/discovered-rank47.json')},
                 generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                 cubin_sha256=hashlib.sha256(Path('build/compiled_scheme.cubin').read_bytes()).hexdigest(),
                 flags=['-O3','-std=c++17','-arch=sm_120','-cubin'])
    Path('results/compiled-build.json').write_text(json.dumps(build,indent=2)+'\n')


if __name__=='__main__':main()
