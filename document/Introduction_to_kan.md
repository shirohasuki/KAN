# KAN网络介绍

**Kolmogorov-Arnold 表示定理**指出，如果 \(f\) 是定义在有界区域上的一个多元连续函数，那么它可以表示为有限个一元连续函数和二元加法运算的组合。更具体地，对于一个光滑的函数 \(f : [0,1]^n \to \mathbb{R}\)，可以表示为：

$$
f(x) = f(x_1,...,x_n)=\sum_{q=1}^{2n+1}\Phi_q\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right)
$$

其中，\(\phi_{q,p}:[0,1]\to\mathbb{R}\)，\(\Phi_q:\mathbb{R}\to\mathbb{R}\)。从某种意义上来说，他们证明了加法是唯一真正的多元函数，因为任何其他函数都可以用一元函数和加法来表示。然而，这种具有 \(2\)-层宽度 \((2n+1)\) 的 Kolmogorov-Arnold 表示可能由于其表达能力有限而无法完全光滑。我们通过将其推广到任意深度和宽度来增强其表达能力。

Kolmogorov-Arnold 表示可以用矩阵形式表示为：

$$
f(x)={\bf \Phi}_{\rm out}\circ{\bf \Phi}_{\rm in}\circ {\bf x}
$$

其中：

$$
{\bf \Phi}_{\rm in}= 
\begin{pmatrix} 
\phi_{1,1}(\cdot) & \cdots & \phi_{1,n}(\cdot) \\ 
\vdots & & \vdots \\ 
\phi_{2n+1,1}(\cdot) & \cdots & \phi_{2n+1,n}(\cdot) 
\end{pmatrix},
\quad 
{\bf \Phi}_{\rm out}=
\begin{pmatrix} 
\Phi_1(\cdot) & \cdots & \Phi_{2n+1}(\cdot)
\end{pmatrix}
$$

我们注意到，\({\bf \Phi}_{\rm in}\) 和 \({\bf \Phi}_{\rm out}\) 都是以下函数矩阵 \({\bf \Phi}\) 的特殊情况（输入维度为 \(n_{\rm in}\)，输出维度为 \(n_{\rm out}\)），我们称其为 Kolmogorov-Arnold 层：

$$
{\bf \Phi}= 
\begin{pmatrix} 
\phi_{1,1}(\cdot) & \cdots & \phi_{1,n_{\rm in}}(\cdot) \\ 
\vdots & & \vdots \\ 
\phi_{n_{\rm out},1}(\cdot) & \cdots & \phi_{n_{\rm out},n_{\rm in}}(\cdot) 
\end{pmatrix}
$$

其中 \({\bf \Phi}_{\rm in}\) 对应 \(n_{\rm in}=n, n_{\rm out}=2n+1\)，而 \({\bf \Phi}_{\rm out}\) 对应 \(n_{\rm in}=2n+1, n_{\rm out}=1\)。

定义了该层后，我们可以通过堆叠多层构造一个 Kolmogorov-Arnold 网络！假设我们有 \(L\) 层，其中第 \(l\) 层 \({\bf \Phi}_l\) 的形状为 \((n_{l+1}, n_{l})\)。那么整个网络可以表示为：

$$
{\rm KAN}({\bf x})={\bf \Phi}_{L-1}\circ\cdots \circ{\bf \Phi}_1\circ{\bf \Phi}_0\circ {\bf x}
$$

相比之下，多层感知机（MLP）的层间交替由线性变换 \({\bf W}_l\) 和非线性激活函数 \(\sigma\) 组成：

$$
{\rm MLP}({\bf x})={\bf W}_{L-1}\circ\sigma\circ\cdots\circ {\bf W}_1\circ\sigma\circ {\bf W}_0\circ {\bf x}
$$

Kolmogorov-Arnold 网络（KAN）可以很容易地可视化：(1) 一个 KAN 仅仅是多层 KAN 层的堆叠。(2) 每个 KAN 层可以看作一个全连接层，其中每条边上都有一个一维函数。以下是一个示例。


**后续还将更新对于该理论推导的探讨，以及其不足**