# Harmonic oscillator SPGPE 
A minimal implementation of SPGPE in 1D harmonic oscillator basis. 

# Usual stuff...

# Projection into aux modes
Aux projection is a general technique that allows change of basis tricks. We _may_ be able to use this to build a composite transform that allows many quadrature operations to happen on a single quadrature rule grid. E.g. when the highest rule is 6-field, and there are also 2-field and 4-field terms, can we put all on the 6-field rule, using 2 additional composite transforms?

# $a$-weighted term
The general problem of mismatched quadrature is one of improper weight. 

Consider a term of the form 

$$
V(x)=e^{-ax^2/2}V_N(x),
$$

where $V_N(x)$ is polynomial of degree $N$. Then we want to put this on a particular quadrature grid.

We can project onto auxilliary modes with frequency $a$ defined as 
$$
\psi_n^a(x)\equiv a^{1/4}\psi_n(x\sqrt{a})=e^{-ax^2/2}\Phi_n^a(x)
$$
where $\Phi_n^a(x)=a^{1/4}P_n(\sqrt{a}x)$ is the polynomial part of the Hermite modes and $P_n(x)$ is the normalized Hermite polynomial of degree $n$ with unit frequency.

For fixed upper degree $N$, the coefficients
$$
d_n \equiv \int dx\;\psi_n^a(x)^*V(x)=\int dx\; e^{-ax^2}\Phi_n^a(x)^*V_N(x)
$$
can be calculated exactly using a quadrature rule, after a change of variables. Changing to $y=x\sqrt{a}$ gives
$$
d_n = \int dy\;\frac{e^{-y^2}}{\sqrt{a}}\Phi_n^a(\frac{y}{\sqrt{a}})^*V_N(\frac{y}{\sqrt{a}})=\sum_i \frac{w_i}{\sqrt{a}}\Phi_n^a(\frac{y_i}{\sqrt{a}})^*V_N(\frac{y_i}{\sqrt{a}})\\
=\sum_i \underbrace{\frac{w_ie^{y_i^2}}{\sqrt{a}}}_{\equiv \tilde w_i}\psi_n^a(\frac{y_i}{\sqrt{a}})^*V(\underbrace{\frac{y_i}{\sqrt{a}}}_{\tilde x_i})= \sum_i \tilde w_i\psi_n^a(\tilde x_i)^*V(\tilde x_i)
$$
So given $V(x)$ on the $\tilde x_i$ grid, can project to aux modes. 

Reconstruction (synthesis) is effected in the usual way on any grid:
$$
V(x) = \sum_n d_n \psi_n^a(x)\tag{test}
$$
providing a simple test.

# Fourier transform
Given the $d_n$ coefficients, can construct in $k$-space as 
$$
\tilde V(k)=\sum_n d_n\phi_n^a(k)
$$
and invert with 
$$
d_n = \int dk\; \phi_n^a(k)^*\tilde V(k),\tag{test}
$$
another simple test. A 2-field quadrature will accomplish the projection integral. (TODO-finish notes)

# Project from different $n$-field grid
Say a particular change of variables has been effected to put a term on a convenient quadrature grid. E.g. 
$$
V'(x)=V(\frac{x}{\sqrt{b}})
$$
is the term $V(x)=e^{-ax^2}V_N(x)$ already on a particular grid:
$$
V'(x)= e^{-ax^2/2b}V_N(x/\sqrt{b}).
$$
Provided the quadrature rule is high enough, we can project onto auxiliary modes my multiplying by 
$$
\psi_n^c(x)=e^{-cx^2/2}\Phi_n^c(x),
$$ 
where $c$ is chosen to put the term into quadrature rule form with respect to integration over $x$: $(a/b + c)/2=1$, or 
$$
c=2-a/b,
$$
with the obvious condition that $a/b<2$ (implications for nonlinear order?).
The projection integral onto the $n$-th mode, _using the same grid_ then reads
$$
d_n = \int dx\; e^{-x^2}\Phi_n^c(x) V_N(\frac{x}{\sqrt{b}})=\sum_i w_i \Phi_n(x_i)V_N(\frac{x_i}{\sqrt{b}}),\\
=\sum_i w_i e^{x_i^2} \psi_n(x_i)V(\frac{x_i}{\sqrt{b}})=\sum_i \tilde w_i \psi_n(x_i)V(\frac{x_i}{\sqrt{b}}),
$$
where again we have effective weights $\tilde w_i=w_ie^{x_i^2}$. As usual, synthesis just involves taking the linear superposition of these same auxiliary modes
$$
V'(x)=\sum_n d_n \psi_n^c(x).\tag{test}
$$
We can take this term to Fourier space via 
$$
\tilde V'(k)=\sum_n d_n\phi_n^c(k),
$$
and invert via 
$$
d_n=\int dk\;\phi_n^c(k)^*\tilde V'(k).\tag{test}
$$
