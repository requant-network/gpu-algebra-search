"""Bind the executed CUDA binary and raw evidence to compiler/source metadata."""
import hashlib,json,pathlib,subprocess
sources=['collect_build.py','cuda/search.cu']
results=['gpu-study.json','gpu-replication.json','transition-validation.json']
record=dict(nvcc=subprocess.check_output(['/usr/local/cuda-13.3/bin/nvcc','--version'],text=True).strip(),
    flags=['-O3','-std=c++17','-arch=sm_120','-cubin'],
    source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},
    result_sha256={p:hashlib.sha256((pathlib.Path('results')/p).read_bytes()).hexdigest() for p in results},
    cubin_sha256=hashlib.sha256(pathlib.Path('build/search.cubin').read_bytes()).hexdigest(),
    scope='Post-run provenance for the exact CUDA binary used by both campaigns. Binary hashes identify measured builds; different rebuilds need not be byte-identical.')
for name in results[:2]:assert json.loads((pathlib.Path('results')/name).read_text())['cubin_sha256']==record['cubin_sha256']
pathlib.Path('results/build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
