# lda & simplex geometry  
enabling a deep understanding and efficient implementation of lda with a focus on the geometric intuition of high-dim simplexes.

## highlights
1. **probabilistic modeling**:
   - e2e lda implementation using **gibbs sampling** & **variational inference**.
   - precise handling of **dirichlet-multinomial conjugacy** for efficient posterior updates.

2. **simplex viz**:
   - rendering low-dim (2d/3d) simplexes & their facets.
   - intuitive viz of topic-word distributions & doc clusters.

3. **high-dim geometry**:
   - constructing & analyzing high-dim simplexes.
   - apps in **topic modeling** & **clustering** with sparse, interpretable dists.

4. **stats insights**:
   - deep dive into **bayesian inference** & **exp family dists**.
   - handling **hypers** (\( \alpha, \beta \)) for dirichlet & multinomial models.

5. **apps**:
   - **doc clustering**: lda applied to organize and interpret unstructured text corpora.
   - **topic alignment**: analyzing high-dim word dists via simplex geometry.
   - **dim reduction**: interpreting lda results using **pca** & **t-sne**.

## structure
- `code/`: lda & simplex computations.
- `docs/`: derivations & theory for lda + geometry.
- `examples/`: sample lda workflows.
- `notebooks/`: interactive hands-on experiments.
- `data/`: example datasets for lda & clustering.
- `tests/`: unit tests for model validation.

## usage
### setup
clone the repo & install deps:
```bash
git clone https://github.com/your-username/lda-simplex-geometry.git
cd lda-simplex-geometry
pip install -r requirements.txt
```

### lda
run lda on sample text data:

```bash
python code/lda_gibbs_sampling.py --data data/sample_docs.txt --topics 3
```

### simplex viz
render a 3-simplex:

```python
from simplex_viz import render_simplex
render_simplex(dim=3, labels=["topic a", "topic b", "topic c"])
```

### notebooks
explore lda interactively:

```bash
jupyter notebook notebooks/
```

## refs
- dirichlet & multinomial conjugacy in bayesian modeling.
- geometric representations in text analysis.
