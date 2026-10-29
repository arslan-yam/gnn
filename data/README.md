# Real-graph benchmarks (downloaded 28 Oct 2026)

Untrusted third-party data. Read only with `np.load(..., allow_pickle=False)` and `torch.load(..., weights_only=True)`;
keep code out of `raw/`.

| Dataset | File | Source | Bytes | SHA-256 |
|---|---|---|---|---|
| Roman-empire | `raw/roman_empire/roman_empire.npz` | https://github.com/yandex-research/heterophilous-graphs/raw/main/data/roman_empire.npz | 20,401,489 | `a58ba741d123bf892fe5c872138d07463d75a2e9012360b8dd78ac2d4766d428` |
| Amazon-ratings | `raw/amazon_ratings/amazon_ratings.npz` | https://github.com/yandex-research/heterophilous-graphs/raw/main/data/amazon_ratings.npz | 27,744,018 | `4c3a3e3b9d9f6cba0fede4625a00aad8c5721c1a36ed771367f446763241c7dd` |
| Peptides-func | `raw/peptides_func/peptidesfunc.zip` | https://www.dropbox.com/s/ycsq37q8sxs1ou8/peptidesfunc.zip?dl=1 (LRGB, as used by PyG `LRGBDataset`) | 29,164,852 | `be12f754516746e77f16d62c4e3aca7f0c96aed90d2a81e7d87cba68951ad21a` |

Extracted from the zip into `raw/peptides_func/peptidesfunc/` (1.0 GB):

| File | SHA-256 |
|---|---|
| `train.pt` | `f79896a52fd153d75925d5975b7298b57bdf72f0250d9e4bddf1f976174f1a6e` |
| `val.pt` | `14eb498022cefbabb7ca32a8220f0997469f6be40c438c3a1f94d1f1595c5b0f` |
| `test.pt` | `49943419a7c3c301ccceeab76adbb3e84d9881920c6574eff573a57eb990e28d` |

## Contents (checked on load)

- **Roman-empire** (Platonov et al. 2023): 22,662 nodes, 32,927 edges, 300 float32 features, 18 classes, edge homophily 0.047.
  `edges` lists each edge once (no reverse copies, no self-loops): symmetrise before use. 10 fixed splits (`train/val/test_masks`,
  shape 10 x n), 50/25/25.
- **Amazon-ratings**: 24,492 nodes, 93,050 edges (stored once each), 300 features, 5 classes, edge homophily 0.380. Same split format.
- **Peptides-func** (LRGB, Dwivedi et al. 2022): 10,873 / 2,331 / 2,331 graphs; each item is a tuple
  `(x [n, 9] int64 atom features, edge_attr [m, 3] int64, edge_index [2, m] int64, y [1, 10] float32)`. 151 nodes per graph on
  average (max 444). 10-label multi-label classification, metric: average precision.
