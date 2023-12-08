# Notation

$$
\mathbf{X} = \left[ \begin{array}{ccc}
\vdots & \vdots &  & \vdots \\
x^{(1)} & x^{(2)} & \cdots & x^{(n)} \\
\vdots & \vdots &  & \vdots \\
\end{array} \right]
\left. \begin{array}{c}
\\
\\
\\
\\
\end{array} \right\} n_x
$$

$$
\begin{array}{c}
\\
\end{array}
\hspace{-2em} % Adjust the -2em value to your preference
\underbrace{\hspace{9em}}_{m}
$$

$\mathbf{X}$ is a set of feature vectors.

- $\mathbf{X} \in \mathbb{R}^{n_{x} \times m}$
- ```
    # python
    X.shape = (n_x, m)
  ```

$n_{x}$ represents the dimensions of an input feature. In a picture example, it represents each pixel of an image.

$m$ denotes the number of training examples. In our case, the number of pictures.

---

$$\mathbf{Y} = [y^{(1)}, y^{(2)}, \dots, y^{(m)}]$$

How about the output **label**?

$\mathbf{Y}$ are the lables and in our example it represents wether the picture in the index is a `1 (cat)` or `0 (non cat)`

- $\mathbf{Y} \in \mathbb{R}^{1 \times m}$
- ```
    # python
    X.shape = (1, m)
  ```

---

In binary classification, the goal is to develop a model that takes an input, often represented as a feature vector $x$, and predicts a binary output $y$. This output typically represents two classes or categories, commonly labeled as `1` or `0`.
