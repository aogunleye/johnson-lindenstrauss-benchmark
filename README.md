# Empirical validation of the Johnson-Lindenstrauss lemma on text data

## Introduction

While studying how LLMs work, I came across this very fascinating lemma and decided to look further.

What caught my attention is the counter-intuitive nature of the JL lemma. The fact that an ultra high dimensional space can be compressed into a wayyy smaller space while preserving geometric distances is what motivated this benchmark. Thought it would be interesting to "verify" it by myself. 

This report presents an empirical evaluation of dimensionality reduction on high-dimensional text data (here, $N=1000$ documents from the `20newsgroups` dataset). By using `scikit-learn` library on Python, we investigate the tradeoff between distance distortion and target dimension $k$, alongside a computational performance comparison between PCA and random projection techniques (Gaussian and Sparse).

---

## 1. Mathematical Framework

![alt text](jllemma.png)

The Johnson-Lindenstrauss lemma guarantees that a set of $N$ points in a high-dimensional space $\mathbb{R}^D$ can be mapped into a much lower-dimensional space $\mathbb{R}^k$ while preserving pairwise euclidean distances up to a factor of $(1 \pm \epsilon)$.

For any $\epsilon \in (0, 1)$ and integer $N$, if the target dimension $k$ satisfies:

$$k \ge \frac{8 \ln(N)}{\epsilon^2}$$

then there exists a linear map $f: \mathbb{R}^D \to \mathbb{R}^k$ such that for all $u, v \in X$:

$$(1 - \epsilon) \Vert{}u - v\Vert{}^2 \le \Vert{}f(u) - f(v)\Vert{}^2 \le (1 + \epsilon) \Vert{}u - v\Vert{}^2$$

In theoretical literature, this bound is often stated using asymptotic notation as $k = O(\log(N)/\epsilon^2)$, showing how $k$ scales with $N$. But in practice, an explicit constant is required for the computer. Applying standard concentration bounds (such as Chernoff/sub-Gaussian bounds) to guarantee that failure probability over all $\binom{N}{2}$ pairs stays strictly below 1 yields the explicit constant factor of 8.

While Johnson and Lindenstrauss (1984) established this upper bound, Larsen and Nelson (2017) proved that this asymptotic scaling is strictly optimal—no algorithm can achieve a target dimension lower than $\Omega(\log(N)/\epsilon^2)$ without exceeding distortion $\epsilon$.

*With the mathematical foundations established, let's now evaluate these properties on real text data.*

---

## 2. Experimental Setup

The benchmark evaluates two primary aspects:

1. **Distortion Decay (Experiment 1):** Measuring empirical distance distortion $\epsilon$ across target dimensions $k \in \{10, 20, 50, 100, 250, 500, 1000, 2000, 4000, 8000\}$.
2. **Computational Efficiency (Experiment 2):** Measuring wall-clock execution times across varying vocabulary sizes $D \in \{1000, 2500, 5000, 10000, 25000, 50000\}$ with $k=100$.

```
Dataset: 20newsgroups (TF-IDF Vectorization)
Sample Size (N): 1,000 documents
Metrics: 95th percentile relative distance error vs. Execution time (seconds)

```

---

## 3. Results & Discussion

### Experiment 1: Distortion vs. Target Dimension $k$

Distance error is calculated as the relative difference between pairwise Euclidean distances in the projected space versus the original TF-IDF space:

$$\text{Error}_{i,j} = \left\vert{} \frac{\Vert{}f(x_i) - f(x_j)\Vert{}}{\Vert{}x_i - x_j\Vert{}} - 1 \right\vert{}$$

| Target Dimension ($k$) | Measured $\epsilon$ (95th Percentile) | Theoretical Bound $\epsilon_{\text{theoretical}}$ |
| --- | --- | --- |
| **10** | High distortion ($\approx 1.8$) | $2.35$ |
| **100** | Acceptable distortion ($\approx 0.5$) | $0.74$ |
| **1000** | Low distortion ($\approx 0.15$) | $0.23$ |
| **8000** | Minimal distortion ($< 0.05$) | $0.08$ |

**Key Observation:** The empirical 95th percentile distortion strictly follows the theoretical $1/\sqrt{k}$ decay curve. The empirical values consistently sit slightly below the theoretical bound, demonstrating that the JL bound provides a safe upper limit for real-world text data.

---

### Experiment 2: Execution Time vs. Original Dimension $D$

| Original Dimension ($D$) | PCA Time (s) | Gaussian RP Time (s) | Sparse RP Time (s) |
| --- | --- | --- | --- |
| **1,000** | $\approx 0.08$ | $\approx 0.01$ | $\approx 0.01$ |
| **10,000** | $\approx 0.50$ | $\approx 0.05$ | $\approx 0.06$ |
| **50,000** | $\approx 2.50$ | $\approx 0.25$ | $\approx 0.30$ |

**Key Observation:**

* **PCA Complexity:** Scaling PCA requires computing singular value decompositions, resulting in an execution time that grows steeply with $D$.
* **Random Projection Efficiency:** Both Gaussian and Sparse Random Projections maintain near-linear scalability, executing approximately **$10\times$ faster** than PCA at $D=50\,000$.

---

## 4. Conclusion

1. **Validity:** The JL Lemma holds empirically on sparse TF-IDF text matrices, showing robust distance preservation without requiring data-dependent fitting.
2. **Scalability:** For high-dimensional text datasets ($D \gg 10\,000$), Random Projection provides an order-of-magnitude speedup over PCA at the cost of a controllable distortion factor $\epsilon$.