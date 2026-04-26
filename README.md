# SPEANN — Synaptic Plasticity Enabled Artificial Neural Network  
![Status](https://img.shields.io/badge/status-experimental-lightgrey)  
![Year](https://img.shields.io/badge/origin-2024-blue)  
![Type](https://img.shields.io/badge/modeling-research-orange)
![Archived](https://img.shields.io/badge/status-archived-red)

---

## 📌 Introductory Note

This repository contains an **old experimental project (late 2024)**.  
Its purpose was to explore ideas related to **artificial synaptic plasticity** in a fully connected neural network, using discretization techniques inspired by biological spiking dynamics.

The project is **not maintained**, not intended for production, and should be considered a **mathematical and conceptual exploration** rather than a functional framework.

This project is experimental and not maintained, but you may still run the simulations using:

```bash
python -m speann
```

The rest of this document preserves the original mathematical formulation and model description.

---

## 🧠 Scientific Summary

SPEANN investigates whether a fully connected artificial neural network can incorporate:

- **Distance‑dependent synaptic delays**  
- **Discrete transformations** mimicking spiking behavior  
- **Voltage traces** through low‑pass filtering  
- A **stochastic synaptic plasticity mechanism** inspired by the TagTriC model  
- A propagation rule closer to **spiking neural networks (SNNs)** than to classical ANNs  

The model introduces:

- A discretized spiking transformation  
- A depleted‑spike propagation mechanism  
- Dendritic trace accumulation  
- Early and late LTP/LTD processes  
- A probabilistic evolution of synaptic weights  

This repository documents the mathematical foundations of this exploratory model.

*(Full mathematical content preserved below — unchanged from the original document.)*

---

## SPEANN: A Synaptic Plasticity Enabled Artificial Neural Network

===
```
---
title: "SPEANN: A Synaptic Plasticity Enabled Artificial Neural Network"
description: "We introduce an artificial synaptic plasticity mechanism in a fully connected artificial neural network, leveraging practical discretization transformations."
date: 2025-04-28
---
```
**keywords**: Synaptic Plasticity; Neural Network; Discretization

---

### Introduction
In this paper, we introduce an artificial synaptic plasticity mechanism in a fully connected artificial neural network, leveraging practical discretization transformations.

### Results
ToDo

### Discussion
ToDo

### Methods

We consider $N$ neurons ($N \in \mathbb{N}^*$), described at all time $t$ by a state vector $\underline{\textbf{X}}(t)$, where:

$$
\begin{array}{c c c c}
\underline{\textbf{X}}: & \mathbb{R}   & \to     & \mathbb{Z}_2^N \\
                        & t            & \mapsto & \underline{\textbf{X}}(t)
\end{array}
$$

We denote by $x^i(t)$ ($i \in \{0,... N-1\}$) the coordinates of $\underline{\textbf{X}}(t)$ in the canonical basis of $\mathbb{Z}_2^N$, and $x_i(t)$ its dual counterpart, which both describe the state of neuron $i$ at time $t$. This naturally defines the coordinate functions $x^i$ and $x_i: \mathbb{R} \to \mathbb{Z}_2$. For all $t<0$, we impose $\underline{\textbf{X}}(t)={\textbf{0}}$.

The neural network is subject to external stimulations, represented at all time $t$ by an input vector $\underline{\textbf{S}}(t) \in \mathbb{Z}_2^N$ with $(s_i)_i$ and $(s^i)_i$ following similar notations as above. For all $t<0$, we also impose $\underline{\textbf{S}}(t)=\textbf{0}$.

The postsynaptic voltage at neuron $j$ at time $t$ is represented by $u_j(t) \in \mathbb{R}$. This defines a potential vector $\underline{\textbf{U}}(t)$ at all time $t$. Again, for all $t<0$, we set $\underline{\textbf{U}}(t)=\textbf{0}$.

We also introduce the $N \times N$ neuronal distances matrix (or mapping matrix):

$$
\underline{\underline{D}} =
\begin{vmatrix}
    d^{0}_{0}   &  \cdots &  d^{0}_{N-1}   \\
    \vdots      &  \ddots &  \vdots        \\
    d^{N-1}_{0} &  \cdots &  d^{N-1}_{N-1} \\
\end{vmatrix}
$$

where, for all couple of neurons $(i, j)$, $d^i_j \in \mathbb{R}_+$ represents the distance between the 2 neurons. Hence, the neuronal distances matrix is symmetrical, and $\forall i, d^i_i=0$.

For any neuron $i$, we also introduce its axonal celerity $c_i(t) \in \mathbb{R^*_+}$ at any time $t$. We impose the initialization constraint $c_i(0)=c_0$, where $c_0$ is a constant. $c_0$ represents the signal celerity in the absence of myelination of the axon. Hence, we have $\forall t \geq 0, c_i(t) \geq c_i(0)=c_0$. We define the signal delay matrix at any time $t$:

$$
\underline{\underline{\Theta}}(t)=
\begin{vmatrix}
    \theta^{0}_{0}(t)   &  \cdots &  \theta^{0}_{N-1}(t)   \\
    \vdots              &  \ddots &  \vdots                \\
    \theta^{N-1}_{0}(t) &  \cdots &  \theta^{N-1}_{N-1}(t) \\
\end{vmatrix}
$$

where, for all couple of neurons $(i, j)$, $\theta^i_j(t)=d^i_j/c_i(t)$ represents the time delay of a signal running from $i$ to $j$. We define $\theta_0=\displaystyle \max_{i,j}(\theta^i_j(0))$. Since $c_i(t) \geq c_i(0)$, we have $\theta^i_j(t) \leq \theta^i_j(0) \leq \theta_0$. Note that contrary to $\underline{\underline{D}}$,  $\underline{\underline{\Theta}}(t)$ is not necessarily symmetrical, and depends on time $t$ (but its diagonal is still filled with zeros).

We define the weight matrix as an $N \times N$ matrix, with values in $\mathbb{R}$ to represent the weights of presynaptic neurons $i$ on postsynaptic neurons $j$ at any time $t$:

$$
\underline{\underline{W}}(t)=
\begin{vmatrix}
    w^{0}_{0}(t)   &  \cdots &  w^{0}_{N-1}(t)   \\
    \vdots         &  \ddots &  \vdots           \\
    w^{N-1}_{0}(t) &  \cdots &  w^{N-1}_{N-1}(t) \\
\end{vmatrix}
$$

With the above notations, we define the continuous propagation equation:

$$
\forall j \in \{ 0,... N-1 \}, \forall t \in \mathbb{R}:
\left\lbrace
    \begin{array}{rcl}
        u_j(t)     &=& s_j(t) + \displaystyle \sum_{i=0}^{N-1}{
            w^i_j(t)
            \cdot x_i( t-\theta^i_j(t) )
            \cdot e^{ -\theta^i_j(t) / \tau_A }
        }
        x_j(t) &=& H(u_j(t) - u^f)
    \end{array}
\right .
$$

where $\tau_A > 0$ is the axon depletion time constant, $u^f$ is the firing voltage constant and $H$ is the Heaviside step function:

$$
\forall z \in \mathbb{R}:
H(z) = \mathbb{1}_{\geq 0}(z) =
\left\lbrace
    \begin{array}{r l}
        1, & z \geq 0 \\
        0, & z < 0
    \end{array}
\right .
$$

Note that the weights $w^i_j$ are applied to the delayed and depleted signals $x_i(t-\theta^i_j(t)) \cdot e^{-\theta^i_j(t) / \tau_A}$ received at postsynaptic neurons $j$ from presynaptic neurons $i$. Note that this is actually *not* a propagation equation since it is missing a time evolution equation (such as a differential equation in continuous conditions). For this reason, we will actually modify the equation. In order to make it easily computable, we will use a discretization technique. The following transformations will also make the modelization more compatible with the spiking nature of natural neural networks.








**Discretization**

We consider a basis time step $\epsilon \in \mathbb{R^*_+}$. For any $n \in \mathbb{Z}$, we define the ticking time $t_n=n\epsilon$.

To describe the action potential, we define $k_S$ and $k_{R}$, both in $\mathbb{N}^*$ with $k_S < k_{R}$: $\epsilon k_S$ and $\epsilon k_{R}$ represent the spike lasting time and the whole action potential duration (including the spike and the refractory period) respectively.

We introduce 3 convenient transformations:
- First, the *Spiking Transformation* (**ST**, denoted with a $\hat{hat}$ in our notations), leading to the spiking state vector $\underline{\hat{\textbf{X}}}$, where, for any neuron $i$:

$$
\forall n \in \mathbb{Z}:
%\left \{
\begin{array}{r c l}
%		\hat{x}_i(0)   & = & x_i(0), \\
    \hat{x}_i(n) & = & x_i(t_n) \cdot \prod_{k=1}^{\min(k_{R}, n)}{(1-\hat{x}_i(n-k))}
\end{array}
%\right .
$$

- Second, the *Depleted Spike Transformation* (**DST**, denoted with a $\tilde{tilde}$ in our notations), leading to the matrix of received signals $\underline{\underline{\tilde{\textbf{X}}}}$, where, for any couple of neurons $(i, j)$, $i$ being the presynaptic neuron and $j$ the postsynaptic one (with $\eta_A = \lfloor \tau_A /\epsilon \rfloor \in \mathbb{N}$, and $\forall n \in \mathbb{Z}, n^i_j(n)= \lfloor \theta^i_j(t_n) /\epsilon \rfloor$):
		
$$
\begin{array}{l c}
    \forall n \in \mathbb{Z}: & \tilde{x}_i^j(n) = \hat{x}_i(n-n^i_j(n)) \cdot e^{-n^i_j(n)/\eta_A}
\end{array}
$$
		
- Third, the *Low-pass Filter Transformation* (**LFT**, denoted with a $\overline{bar}$), leading to the matrix of dendrite traces, where, for any couple of neurons $(i, j)$, $i$ being the presynaptic neuron and $j$ the postsynaptic one (with $\eta_X \in \mathbb{N}$, and $\epsilon \cdot \eta_X$ being the dendrite persistence characteristic time ($\tau_X$), which represents the depletion of concentration of neurotransmitters left on the dendrite (s.a. [Ca^+]) -- this transformation will be used in the synaptic plasticity model):

$$
\begin{array}{l c}
    \forall n \in \mathbb{Z}: & \overline{x}_i^j(n) = \frac{1}{\eta_X} \displaystyle \sum_{k=0}^{n} \tilde{x}_i^j(k) \cdot e^{-(n-k)/\eta_X}
\end{array}
$$



**Discrete propagation equation**

Now we can replace the previous continuous propagation equation with the following discrete propagation equation:

$$
\forall j \in \{ 0, ... N-1 \}, \forall n \in \mathbb{Z}:\\
    \left\lbrace
    \begin{array}{rcl}
        u_j(t_n)     &=& s_j(t_n) + \displaystyle \sum_{i=0}^{N-1}{w^i_j(t_n) \cdot \overline{x}^j_i}(n) \\
        x_j(t_{n+1}) &=& H(u_j(t_n) - u^f)
    \end{array}
    \right .
$$

Note that this equation introduces 2 key differences with the previous one: first by introducing the propagation with $x_j(t_{n+1})$ in the second line, and second by using the trace $\overline{x}^j_i$ instead of the delayed signal in the first line: the received signals at the dendrites leave voltage traces (e.g. accumulation of neurotransmitters), which are taken into account in the determination of the neuron internal voltage. Remember that $\textbf{S}$ depends only on external stimuli (such as signals from external sensors). Hence, to complete this propagation equation, we need to specify the evolution of the weight matrix $\underline{\underline{\textbf{W}}}(t_n)$.






**The synaptic plasticity model**

To model the time evolution of the weight matrix, we will use a synaptic plasticity model largely inspired by the TagTriC model [[clopath](https://doi.org/10.1371/journal.pcbi.1000248)]. In this modelization, the weights will follow a stochastic process. More precisely, the synaptic weight between presynaptic neuron $i$ and postsynaptic neuron $j$ is given by:

$$
\begin{array}{l c}
    \forall n \in \mathbb{Z}: & w^i_j(t_{n+1}) = w^i_j(t_n) \cdot (1 + \underline{\textbf{A}}^- \cdot \underline{\textbf{Y}}^i_j(t_n) + \beta \cdot z^i_j(t_n))
\end{array}
$$
    
where:
- the early long term potentiation/depression increase/decrease factor $\underline{\textbf{A}}^- \cdot \underline{\textbf{Y}}^i_j(t_n)$ is the dot product of two vectors of $\mathbb{R}^3$: a constant vector $\underline{\textbf{A}}^-$, and a random vector $\underline{\textbf{Y}}^i_j(t_n)$ representing the early long term memory state of the synaptic connection, which can be one of the 3 following values: $\textbf{e}_N$ (neutral state), $\textbf{e}_l$ (low state $l$) and $\textbf{e}_h$ (high state $h$). Using $(\textbf{e}_N, \textbf{e}_l, \textbf{e}_h)$ as basis, the constant vector is (with $\alpha > 0$):

$$
\underline{\textbf{A}}^- = \begin{vmatrix}{0, -\alpha, 1}\end{vmatrix}.
$$

- $\beta \cdot z^i_j(t_n)$ is the late long term potentiation/depression increase factor, with $\beta>0$ a constant, and $z^i_j(t_n)$ the late long term memory state.

To complete the description, we need to describe the evolution of $\underline{\textbf{Y}}^i_j$ and $z^i_j$.




**Early long term potentiation (E-LTP) and early long term depression (E-LTD)**

For all postsynaptic neuron $j$, we define the potentiation low-pass-filtered voltage $\overline{u}^{(P)}_j$, and the depression low-pass-filtered voltage $\overline{u}^{(D)}_j$ by:

$$
\forall n \in \mathbb{Z}: \\
\left\lbrace
\begin{array}{r c l}
        \overline{u}^{(P)}_j(n) &=& \frac{1}{\eta_P} \displaystyle \sum_{k=0}^{n} u_j(t_n-t_k) \cdot e^{-k/\eta_P} \\
        \overline{u}^{(D)}_j(n) &=& \frac{1}{\eta_D} \displaystyle \sum_{k=0}^{n} u_j(t_n-t_k) \cdot e^{-k/\eta_D}
    \end{array}
\right .
$$

where $\eta_P \in \mathbb{N}$ (resp. $\eta_D \in \mathbb{N}$), with $\epsilon \cdot \eta_P$ (resp. $\epsilon \cdot \eta_D$) is the time constant of the potentiation (resp. depression) low-pass filter. Note that this is actually the low-pass transformation of $\underline{\textbf{U}}$, with parameters $\eta_P$ (resp. $\eta_D$).

We define transition rates for the random vector $\underline{\textbf{Y}}^i_j(t_n)$, for a postsynaptic neuron $j$ and a presynaptic neuron $i$, at time $t_n$:

We define the transition matrix of the Markov chain

$$
\underline{\mathbf{Y}}^i_j
$$

as:

$$
\mathbf{P}_{i,j} =
\begin{bmatrix}
\ddots & k_L \epsilon & k_H \epsilon \\
\rho^{(L)}_{i,j} \epsilon & \ddots & 0 \\
\rho^{(H)}_{i,j} \epsilon & 0 & \ddots
\end{bmatrix}
$$

where each entry

$$
\mathbf{P}_{i,j}(a \to b)
$$

denotes

$$
\mathbb{P}(\underline{\mathbf{Y}}^i_j(t_{n+1}) = b \mid \underline{\mathbf{Y}}^i_j(t_n) = a),
$$

and:

$$
\forall n \in \mathbb{Z}: \\
\left\lbrace
\begin{array}{r c l}
        \rho^{(L)}_{i,j} &=& A_L \tilde{x}^j_i(n) [\overline{u}^{(D)}_j(n)-u^L]^+ \\
        \rho^{(H)}_{i,j} &=& A_H \overline{x}^j_i(n) [\overline{u}^{(P)}_j(n)-u^L]^+[u_j(t_n)-u^H]^+
    \end{array}
\right .
$$

and $k_L$, $k_H$, $A_L$, $A_H$, $u^L$, $u^H$ are constants. This defines the evolution $\underline{\textbf{Y}}^i_j$.








**Late long term potentiation (L-LTP) and late long term depression (L-LTD)**

$$
\forall i, j \in \{ 0, ... N-1 \}, \forall n \in \mathbb{Z}: \\
\left\lbrace
    \begin{array}{r c l}
        p_j(n+1) 		&=& p_j(n) \cdot ( 1 - \frac{1}{\eta_p} ) + ( 1 - p_j(n) ) \cdot H( \sum_{i=0}^{N-1}{\langle +| \textbf{Y}^i_j(t_n)\rangle} - {N}_p) \cdot k_p \epsilon	\\
        z^i_j(t_{n+1}) 	&=& z^i_j(t_n) + \frac{1}{\eta_z} (
            Det({{
                z^i_j(t_n) \underline{ \textbf{Id} }_3 - Diag( \underline{ \textbf{A} }^+ )
                }})
        + \gamma p_j(n) \langle -|\textbf{Y}^i_j(t_n)\rangle)
    \end{array}
\right .
$$

where $\eta_p$, $k_p$, $\eta_z$, $\gamma$ and $N_p$ are constants, $H$ the Heaviside function, $\langle +|=\langle 0, 1, 1|$, $\langle -|=\langle 0, -1, 1|$ and $\underline{\textbf{A}}^+=\langle 0, \alpha, 1|$. This defines the evolution of $z^i_j$.

To complete the model description, we need to set the initial conditions for the weights.

**Computation of $w^j_j$ for any neuron $j$**

- Let suppose $n$ such that $t_{n+1}$ is the very first firing time for neuron $j$: $\forall k \leq n, x_j(t_k) = 0$ and $x_j(t_{n+1}) = 1$. We suppose no other external stimulation: $\forall k \in \mathbb{N}, s_j(t_{n+k}) = 0$.

- Let suppose that $\forall k \in \mathbb{N}, \forall i \in \{ 0, ... N-1 \} \setminus \{j\}, \tilde{x}^j_i(n+k)=0$. This hypothesis is actually a case study: it will give us an easy way to compute $w^j_j$.

- Let suppose that $u_j$ remains above the threshold $u^f$ for as long as the spiking time $\epsilon k_S$.

- $\tilde{x}^j_j(n+1) = 1 \implies \overline{x}^j_j(n+1) = 1/\eta_X \implies u_j(t_{n+1})=w^j_j/\eta_X$, with $u_j(t_{n+1})>u^f$;

- ...

- $\tilde{x}^j_j(n+k_S) = 1 \implies \overline{x}^j_j(n+k_S) = \frac{1}{\eta_X} \sum_{k=0}^{k_S-1}{
			e^{-k/\eta_X}
			} \implies u_j(t_{n+k_S}) = \frac{w^j_j}{\eta_X}
			\sum_{k=0}^{k_S-1}{
			e^{-k/\eta_X}
			}$, with
			$u_j(t_{n+k_S})>u^f
			\implies
			u^f < \frac{w^j_j}{\eta_X} \frac{1-e^{-k_S/\eta_X}}{1-e^{-1/\eta_X}}
			\approx w^j_j \frac{k_S}{\eta_X}$.

- Hence, in order to remain consistent with the fact that a spike should last $k_S$ basis periods, we should have $w^j_j \geq w_0$ where we set $w_0=u^f \eta_X \frac{1-e^{-1/\eta_X}}{1-e^{-k_S/\eta_X}} \approx \frac{\eta_X}{k_S}u^f$.





---

## Repository Structure

- [`/speann`](./speann) — Python package  
- [`notebook.ipynb`](./notebook.ipynb) — Example simulations  
- `README.md` -- this document

---

## References

- [Clopath C, Ziegler L, Vasilaki E, Büsing L, Gerstner W (2008) Tag-Trigger-Consolidation: A Model of Early and Late Long-Term-Potentiation and Depression. PLOS Computational Biology 4(12): e1000248. https://doi.org/10.1371/journal.pcbi.1000248](https://doi.org/10.1371/journal.pcbi.1000248)

---

## License
This project is released under the [MIT License](./LICENSE).

---

## Citation

If you use or reference this work, you may cite it informally as:

> U. Tan, *SPEANN: A Synaptic Plasticity Enabled Artificial Neural Network*, experimental project (2024–2025).

---

## Project Status

This project is **archived** and kept online for reference and educational purposes.

---