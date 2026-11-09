"""Check the optional rank-49 compositional starting scheme exactly."""
import json,pathlib,random
from experiments.tensor import evaluate,kronecker_square,multiply,verify
base=json.loads(pathlib.Path('fixtures/alphatensor-2x2x2.json').read_text())['scheme']
scheme=kronecker_square(base,base,2,2);assert len(scheme)==49 and verify(scheme,(4,4,4))
rng=random.Random(8675309)
for _ in range(100):
    a=rng.getrandbits(16);b=rng.getrandbits(16);assert evaluate(scheme,a,b)==multiply(a,b,(4,4,4))
print('Optional rank-49 start: exact tensor and 100 independent products passed')
