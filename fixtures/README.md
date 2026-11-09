# Reference certificates

The `alphatensor-*.json` files contain numerical factorizations converted from
[google-deepmind/alphatensor](https://github.com/google-deepmind/alphatensor),
revision `1949163da3bef7e3eb268a3ac015fd1c2dbfc767`, under Apache-2.0.
Copyright 2022 DeepMind Technologies Limited. The original license is retained
in [ALPHATENSOR-LICENSE](ALPHATENSOR-LICENSE).

The third factor is transposed into the row-major output convention. The
import script records the source archive hash, and our independent exact
tensor verifier checks the resulting certificates. These are attributed
reference algorithms, not discoveries by Requant Network.

`paper-7x7x7-rank245.json` is the numerical Appendix A certificate from
Medley, Gokul, Luu and Manolios, [arXiv:2609.17533v2](https://arxiv.org/html/2609.17533v2),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It was decoded into factor masks and checked with our own verifier. The paper's
program was not executed. This reference is a CPU verification control;
our GPU search campaign concerns 4x4 and 3x4x3 instances.
