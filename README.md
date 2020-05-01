# Deep Stock Ranking

Working code repository for a deep learning stock ranking research project.

## Evaluating data

A universe of n = 2023 stock daily close prices are evaluated over roughly 20 years. For a time series
<img src="http://www.sciweavers.org/tex2img.php?eq=%28s_t%29%20%5Cin%5Cmathbb%7BR%7D%2C%5Cquad%5Ctext%7Bfor%20%7D%201%20%3C%20t%5Cin%5Cmathbb%7BZ%7D%2C&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="(s_t) \in\mathbb{R},\quad\text{for } 1 < t\in\mathbb{Z}," width="337" height="18" />

we calculate log-returns,
$$
(x_t) = \log x_t - \log x_{t-1}.
$$


## Road map

- [x] PyTorch implementation of Neuralsort
- [x] Quandl EOD data import into SQLite3 database
- [ ] Adapt WaveNet to *volatility-innovation* asset pricing model 
- [ ] Implement Quant GAN model to generate financial time series
  - [x] Lambert W transform
  - [x] Preprocess data as in [1].
  - [ ] Quant GAN data loader
  - [ ] Generator model
  - [ ] Discriminator model
- [ ] Evaluate various predictor models for input to Neuralsort using criteria of [6].

### Papers cited

<sup>[1] M. Wiese, R. Knobloch, R. Korn, P. Kretschmer, "Quant GANs: Deep Generation of Financial Time Series," arXiv:1907.06673v2 [q-fin.MF], Dec. 2019.</sup>

<sup>[2] M. Wiese, R. Knobloch, R. Korn, "Copula & Marginal Flows: Disentangling the Marginal from its Join," arXiv:1907.03361v1 [cs.LG], July 2019.</sup>

<sup>[3] G. M. Goerg, "The Lambert Way to Gaussianize heavy tailed data with the inverse of Tukey's h transformation as a special case," arXiv:1010.2265v5 [math.ST], Dec. 2012.</sup>

<sup>[4] A. Grover, E. Wong, A. Zweig, S. Ermon, "Stochastic Optimization of Sorting Networks via Continuous Relaxations," arXiv:1903.08850 [stat.ML]. Apr. 2019</sup>

<sup>[5] A. van den  Oord, et al., "WaveNet: A Generative Model for Raw Audio," arXiv:1609.03499v2 [cs.SD] Sep. 2016.</sup>

<sup>[6] Q. Song, A. Liu, S. Y. Yang, "Stock portfolio selection using learning-to-rank algorithms with news sentiment," *Neurocomputing*, vol. 264, pp. 20-28, Nov. 2017.</sup>

