# distutils: libraries = flint
r"""
The main methods underlying the integral computation in the "radial
Brion" method (using Brion's formula, integrated along the radial
variable): compute the sum of the integrand `G` given by Brion's
formula over the nodes of a conical product rule on the standard
simplex. The mathematical content on which this is based
can be found in ``mathematics/radial-brion.pdf``.

Letting `u_i` be the nodes in the `i`-th axis, `1 \leq i \leq d - 1`,
the rule (:mod:`sieve_integral_brion.rules`) sets `x_i = u_i
\prod_{i' < i} (1 - u_{i'})`, `x_d = \prod_i (1 - u_i)`, with the
weight `\prod_i w_i(u_i)`. The coordinate ``order[i - 1]`` of the node
is `x_i`. The nodes belong to the standard simplex `\Delta \subset \RR^d`.

The integrand at a node `x` is

.. MATH::

    G(x) = \sum_v \sum_{\beta \in S} \sum_{l = 0}^D c_{v, \beta, l}(x) \,
    T_{v, \beta} \sum_{i = 0}^{D - l} q_{l + i} x_s^{l + i}
    \frac{\psi(v)^i}{i!} R(i - |\beta| + \sigma, \ell_v),

where

- `v` runs over the vertices of the polytope, `\ell_v = \langle x,
  v\rangle`, and `\sigma` is `0` for a full-dimensional polytope, `1` for one
  on a hyperplane;

- `c_{v, \beta, l} = \sum_{C \ni v} |\det C| \prod_{g \in C} a_g^{-1}
  Y^C_{\beta, l}`, over the simplicial cones `C` at `v`, with `a_g =
  \langle x, g\rangle` and `Y^C_{\beta, l}` the coefficient of `\zeta^\beta
  \xi^l` in `\prod_{g \in C} (1 - \sum_f \zeta_f \alpha_{f, g} - \xi
  \alpha_{\psi, g})^{-1}`, `\alpha_{f, g} = \langle A_f, g\rangle/a_g`,
  `\alpha_{\psi, g} = \langle \psi_1, g\rangle/a_g`;

- `S` is a set of exponents `\beta` closed under taking smaller exponents,
  `T_{v, \beta} = \partial^\beta \Phi(L(v))` for the weight `\Phi(L(t))`,
  `L_f(t) = \langle A_f, t\rangle + b_f`;

- `Q = \sum_{n = 0}^D q_n y^n` is the factor `Q(\psi(t)/t_s)`, `\psi(t) =
  \psi_0 + \langle \psi_1, t\rangle`;

- `R(a, \ell) = (a - 1)! \ell^{-a}` for `a \geq 1`, and `R(-e, \ell) =
  (-1)^e \ell^e (H_e - \log \ell)/e!` for `e \geq 0`, `H_e` the harmonic
  number.

The cones of each vertex are summed into a buffer of the coefficients
`c_{v, \beta, l}`. We treat each vertex separately, discarding a
vertex's data as we move on to the next vertex. The coefficients `Y^C`
come from the in-place recurrence `Y_{\beta, l} \mathrel{+}= \sum_f
\alpha_{f, g} Y_{\beta - e_f, l} + \alpha_{\psi, g} Y_{\beta, l - 1}`, in
increasing order of `(\beta, l)`, for each ray `g` of `C`.

When a node gives a non-finite value, or a radius above a tolerance, it is
left out of the sum and returned to be computed again at a higher precision;
:meth:`BrionSum.zero_generators` lists the hyperplanes `\langle x, g\rangle
= 0` which may hold at it.

EXAMPLES:

On the rectangle `[1, 2] \times [1, 3]`, divided by its centre `(3/2, 2)`,
the sum approaches `\int dt_1 dt_2/(t_1 t_2) = \log 2 \log 3`::

    sage: from sieve_integral_brion._brion_sum import BrionSum
    sage: from sieve_integral_brion.geometry import vertex_cones, generator_table
    sage: from sieve_integral_brion.rules import conical_factors
    sage: box = Polyhedron(vertices=[(1, 1), (2, 1), (1, 3), (2, 3)])
    sage: vertices, cones = vertex_cones(box, (3/2, 2), False)
    sage: generators, cones = generator_table(cones)
    sage: brion_sum = BrionSum(100, conical_factors([16], 100), vertices,
    ....:                      generators, cones, 0)
    sage: value, failed = brion_sum.run()
    sage: failed, abs(value - log(RBF(2)) * log(RBF(3))) < 10^-12
    ([], True)

AUTHORS:

- Sary Drappeau

AI DISCLOSURE:

Claude Opus 5.5 wrote this module, from a prototype it had written for the
``sieve_integral`` library.
"""
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from sage.libs.flint.types cimport arb_ptr, arb_t, slong
from sage.libs.flint.arb cimport *
from sage.libs.flint.arb_macros cimport arb_radref
from sage.libs.flint.mag cimport mag_cmp_2exp_si
from sage.rings.real_arb cimport RealBall
from sage.rings.real_arb import RealBallField

# the numbers of ball arrays and of integer arrays of a BrionSum
cdef enum:
    _NARB_ARRAYS = 30
    _NINT_ARRAYS = 8


cdef void _set_values(arb_ptr target, values, field):
    # the values, as balls of the field
    cdef RealBall ball
    cdef slong i = 0
    for value in values:
        ball = field(value)
        arb_set(target + i, ball.value)
        i += 1


def _exponent_structure(exponents):
    r"""
    Return, for the exponents `\beta \in S`, the position of each `\beta -
    e_f` (`-1` if `\beta_f = 0`) and `|\beta|`, after checking that `S`
    starts with `0` and holds the smaller exponents before each one.
    """
    position = {tuple(beta): s for s, beta in enumerate(exponents)}
    F = len(exponents[0])
    if len(position) != len(exponents) or any(exponents[0]):
        raise ValueError("the exponents must be distinct and start with 0")
    predecessors = []
    for s, beta in enumerate(exponents):
        if len(beta) != F:
            raise ValueError("the exponents must have the same length")
        row = []
        for f in range(F):
            if beta[f] == 0:
                row.append(-1)
                continue
            p = position.get(tuple(b - (i == f) for i, b in enumerate(beta)), -1)
            if not 0 <= p < s:
                raise ValueError(f"the exponent {tuple(beta)} comes before one of "
                                 "its predecessors, or has none")
            row.append(p)
        predecessors.append(row)
    return predecessors, [sum(beta) for beta in exponents]


cdef class BrionSum:
    r"""
    The sum of `w(x) G(x)` over the nodes `x` of a conical product rule, in
    ball arithmetic at a given precision.

    INPUT:

    - ``prec`` -- the precision of the computation.

    - ``factors`` -- the `d - 1` pairs ``(nodes, weights)`` of the rule,
      each a list of balls (:func:`~sieve_integral_brion.rules.conical_factors`).

    - ``vertices`` -- the `V` vertices, `d` numbers each.

    - ``generators`` -- the rays of the cones up to sign, `d` integers each.

    - ``cones`` -- triples ``(vertex, rays, det)``: the position of the
      vertex, pairs ``(position, sign)`` giving the rays among
      ``generators``, and `|\det C|`
      (:func:`~sieve_integral_brion.geometry.generator_table`). The cones of
      a vertex follow each other.

    - ``shift`` -- `\sigma`.

    - ``order`` (default: the identity) -- the coordinates of the node
      which receive `x_1, \ldots, x_d`.

    - ``exponents`` (default: ``[()]``) -- the exponents `\beta \in S`, each
      after the exponents below it, the first one `0`.

    - ``slopes`` (default: ``None``) -- for each generator, the `\langle A_f,
      g\rangle`; needed when the exponents have positive length.

    - ``derivatives`` (default: all `1`) -- for each vertex, the `T_{v,
      \beta}` in the order of ``exponents``.

    - ``ratio`` (default: ``None``, meaning `Q = 1`) -- a tuple ``(q,
      divisor, psi_slopes, psi_values)``: the `q_n`, the index `s - 1` of the
      divisor `t_s`, the `\langle \psi_1, g\rangle` of the generators, and the
      `\psi(v)` of the vertices.

    - ``tolerance_exponent`` (default: ``None``) -- an integer `e`: the nodes
      whose value has a radius above `2^e` are left out; with ``None``, only
      the nodes whose value is not finite are.
    """
    cdef slong prec, d, k, V, ngen, ncones, rank, F, NS, DQ, M, E
    cdef slong shift, divisor, tolerance_exponent
    cdef bint has_tolerance, need_log
    # the ball arrays, each allocated on its own: their pointers and sizes,
    # and whether each was allocated (an empty one is not), for __dealloc__
    cdef arb_ptr arb_arrays[_NARB_ARRAYS]
    cdef slong arb_array_sizes[_NARB_ARRAYS]
    cdef bint arb_allocated[_NARB_ARRAYS]
    cdef slong narb_arrays
    # the integer arrays, likewise: their pointers, and whether each was
    # allocated, for __dealloc__
    cdef long *int_arrays[_NINT_ARRAYS]
    cdef bint int_allocated[_NINT_ARRAYS]
    cdef slong nint_arrays
    # the rule
    cdef long *msize
    cdef long *offsets
    cdef long *order
    cdef arb_ptr nodes, complements, weights
    # the polytope and the weight
    cdef arb_ptr vertices, generators, slopes, psi_slopes, psi_values, dets
    cdef arb_ptr T, q, factorial, hconst_harm, hconst_fact
    cdef long *cone_vertex
    cdef long *cone_gens
    cdef long *cone_signs
    cdef long *pred
    cdef long *degree
    # the work space of a node: a_g and 1/a_g in a and ia, alpha_{f,g} in y,
    # alpha_{psi,g} in ypsi, c_{v,beta,l} in gamma, Y^C in H
    cdef arb_ptr x, a, ia, y, ypsi, gamma, H, Ssum, zk, qx, R
    cdef arb_ptr ellpow, iellpow, u, Vn, tmp

    def __cinit__(self):
        cdef slong i
        for i in range(_NARB_ARRAYS):
            self.arb_allocated[i] = False
        self.narb_arrays = 0
        for i in range(_NINT_ARRAYS):
            self.int_allocated[i] = False
        self.nint_arrays = 0

    cdef arb_ptr _arb_array(self, slong size) except? NULL:
        # Helper for allocation of an array of size balls. The array is recorded for __dealloc__.
        # returns NULL if empty
        cdef slong i = self.narb_arrays
        if i == _NARB_ARRAYS:
            raise RuntimeError("more ball arrays than _NARB_ARRAYS")
        self.narb_arrays += 1
        self.arb_array_sizes[i] = size
        if size == 0:
            return NULL
        self.arb_arrays[i] = _arb_vec_init(size)
        self.arb_allocated[i] = True
        return self.arb_arrays[i]

    cdef long *_int_array(self, slong size) except? NULL:
        # Same helper for int arrays
        cdef slong i = self.nint_arrays
        if i == _NINT_ARRAYS:
            raise RuntimeError("more integer arrays than _NINT_ARRAYS")
        self.nint_arrays += 1
        if size == 0:
            return NULL
        self.int_arrays[i] = <long *> PyMem_Malloc(size * sizeof(long))
        if self.int_arrays[i] == NULL:
            raise MemoryError
        self.int_allocated[i] = True
        return self.int_arrays[i]

    def __init__(self, slong prec, factors, vertices, generators, cones,
                 slong shift, order=None, exponents=None, slopes=None,
                 derivatives=None, ratio=None, tolerance_exponent=None):
        cdef slong i, j, s, f, idx, l, pos
        cdef long *gens
        cdef long *signs
        cdef long *row
        if self.narb_arrays:
            raise RuntimeError("this BrionSum is already initialised")
        field = RealBallField(prec)
        self.prec = prec
        self.k = len(factors)
        self.d = self.k + 1
        if self.k < 1:
            raise ValueError("the rule needs at least one factor: d >= 2")
        self.V = len(vertices)
        self.ngen = len(generators)
        self.ncones = len(cones)
        self.shift = shift
        if self.ncones == 0:
            raise ValueError("there must be at least one cone")
        self.rank = len(cones[0][1])
        if order is None:
            order = list(range(self.d))
        if sorted(order) != list(range(self.d)):
            raise ValueError(f"the order {order} is not a permutation of "
                             f"range({self.d})")
        if exponents is None:
            exponents = [()]
        predecessors, degrees = _exponent_structure(exponents)
        self.F = len(exponents[0])
        self.NS = len(exponents)
        self.E = max(degrees)
        if self.F and (slopes is None or len(slopes) != self.ngen):
            raise ValueError("the slopes of the generators are needed")
        if derivatives is None and self.NS > 1:
            raise ValueError("the derivatives at the vertices are needed")
        if derivatives is not None and len(derivatives) != self.V:
            raise ValueError("one row of derivatives per vertex is needed")
        if ratio is None:
            self.DQ = 0
            self.divisor = -1
        else:
            self.DQ = len(ratio[0]) - 1
            self.divisor = ratio[1]
        self.M = self.NS * (self.DQ + 1)
        # logarithms appear as soon as some R(a) has a <= 0
        self.need_log = shift - self.E <= 0
        self.has_tolerance = tolerance_exponent is not None
        self.tolerance_exponent = tolerance_exponent if self.has_tolerance else 0
        seen = set()
        for c in range(self.ncones):
            if len(cones[c][1]) != self.rank:
                raise ValueError("the cones must have the same number of rays")
            if cones[c][0] in seen and cones[c][0] != cones[c - 1][0]:
                raise ValueError("the cones of a vertex must follow each other")
            seen.add(cones[c][0])


        # allocation of the arrays of balls
        nodes_count = [len(nodes) for nodes, _ in factors]
        msum = sum(nodes_count)
        self.nodes       = self._arb_array(msum)
        self.complements = self._arb_array(msum)
        self.weights     = self._arb_array(msum)
        self.vertices    = self._arb_array(self.V * self.d)
        self.generators  = self._arb_array(self.ngen * self.d)
        self.slopes      = self._arb_array(self.ngen * self.F)
        self.psi_slopes  = self._arb_array(self.ngen if ratio is not None else 0)
        self.psi_values  = self._arb_array(self.V if ratio is not None else 0)
        self.dets        = self._arb_array(self.ncones)
        self.T           = self._arb_array(self.V * self.NS)
        self.q           = self._arb_array(self.DQ + 1)
        self.factorial   = self._arb_array(self.DQ + 1)
        self.hconst_harm = self._arb_array(self.E + 1)
        self.hconst_fact = self._arb_array(self.E + 1)
        self.x           = self._arb_array(self.d)
        self.a           = self._arb_array(self.ngen)
        self.ia          = self._arb_array(self.ngen)
        self.y           = self._arb_array(self.ngen * self.F)
        self.ypsi        = self._arb_array(self.ngen if self.DQ > 0 else 0)
        self.gamma       = self._arb_array(self.M)
        self.H           = self._arb_array(self.M)
        self.Ssum        = self._arb_array((self.DQ + 1) * (self.E + 1))
        self.zk          = self._arb_array((self.DQ + 1) * (self.E + 1))
        self.qx          = self._arb_array(self.DQ + 1)
        self.R           = self._arb_array(self.E + self.DQ + 2)
        self.ellpow      = self._arb_array(self.E + 1)
        self.iellpow     = self._arb_array(self.DQ + 2)
        self.u           = self._arb_array(self.NS)
        self.Vn          = self._arb_array(self.DQ + 1)
        self.tmp         = self._arb_array(8)

        # allocation of the arrays of integers
        self.msize       = self._int_array(self.k)
        self.offsets     = self._int_array(self.k)
        self.order       = self._int_array(self.d)
        self.cone_vertex = self._int_array(self.ncones)
        self.cone_gens   = self._int_array(self.ncones * self.rank)
        self.cone_signs  = self._int_array(self.ncones * self.rank)
        self.pred        = self._int_array(self.M * (self.F + 1))
        self.degree      = self._int_array(self.M)

        # the rule
        pos = 0
        for i, (nodes, weights) in enumerate(factors):
            self.msize[i] = len(nodes)
            self.offsets[i] = pos
            _set_values(self.nodes + pos, nodes, field)
            _set_values(self.weights + pos, weights, field)
            for j in range(len(nodes)):
                arb_one(self.complements + pos + j)
                # Complements will be needed when computing the
                # coordinates of points x in the simplex, to be used
                # in the actual computation of G(x)
                arb_sub(self.complements + pos + j, self.complements + pos + j,
                        self.nodes + pos + j, prec)
            pos += len(nodes)
        for i in range(self.d):
            self.order[i] = order[i]

        # the polytope
        for i in range(self.V):
            _set_values(self.vertices + i * self.d, vertices[i], field)
        for i in range(self.ngen):
            _set_values(self.generators + i * self.d, generators[i], field)
            if self.F:
                _set_values(self.slopes + i * self.F, slopes[i], field)
        for c in range(self.ncones):
            self.cone_vertex[c] = cones[c][0]
            gens = &self.cone_gens[c * self.rank]
            signs = &self.cone_signs[c * self.rank]
            for i, (g, sign) in enumerate(cones[c][1]):
                gens[i] = g
                signs[i] = sign
        _set_values(self.dets, [c[2] for c in cones], field)

        # the weight
        for i in range(self.V):
            if derivatives is None:
                arb_one(self.T + i)
            else:
                _set_values(self.T + i * self.NS, derivatives[i], field)
        if ratio is None:
            arb_one(self.q)
        else:
            _set_values(self.q, ratio[0], field)
            _set_values(self.psi_slopes, ratio[2], field)
            _set_values(self.psi_values, ratio[3], field)

        # idx = s (DQ + 1) + l: its predecessors along the zeta_f, then
        # along xi, -1 if there is none
        for idx in range(self.M):
            s = idx // (self.DQ + 1)
            l = idx % (self.DQ + 1)
            self.degree[idx] = degrees[s]
            row = &self.pred[idx * (self.F + 1)]
            for f in range(self.F):
                p = predecessors[s][f]
                row[f] = p * (self.DQ + 1) + l if p >= 0 else -1
            row[self.F] = idx - 1 if l > 0 else -1

        # (a - 1)! for a = 1 .. DQ + 1; (-1)^e/e! and H_e for e = 0 .. E
        arb_one(self.factorial)
        for i in range(1, self.DQ + 1):
            arb_mul_si(self.factorial + i, self.factorial + i - 1, i, prec)
        arb_one(self.hconst_fact)
        arb_zero(self.hconst_harm)
        for i in range(1, self.E + 1):
            arb_div_si(self.hconst_fact + i, self.hconst_fact + i - 1, -i, prec)
            arb_one(self.hconst_harm + i)
            arb_div_si(self.hconst_harm + i, self.hconst_harm + i, i, prec)
            arb_add(self.hconst_harm + i, self.hconst_harm + i,
                    self.hconst_harm + i - 1, prec)

    def __dealloc__(self):
        cdef slong i
        for i in range(self.narb_arrays):
            if self.arb_allocated[i]:
                _arb_vec_clear(self.arb_arrays[i], self.arb_array_sizes[i])
        for i in range(self.nint_arrays):
            if self.int_allocated[i]:
                PyMem_Free(self.int_arrays[i])

    @property
    def nodes_count(self):
        r"""
        The numbers of nodes of the factors of the rule.
        """
        return tuple(self.msize[i] for i in range(self.k))

    cdef void _vertex_term(self, slong v, arb_ptr value):
        # adds the term of the vertex v, from its coefficients in self.gamma
        # also assumes that self.qx was computed (in _integrand)
        cdef slong i, e, l
        cdef slong prec = self.prec
        cdef arb_ptr t = self.tmp
        cdef arb_ptr ell = self.tmp + 1
        cdef arb_ptr logell = self.tmp + 2
        cdef arb_ptr inv = self.tmp + 3
        cdef arb_ptr z = self.tmp + 4
        cdef arb_ptr total = self.tmp + 5
        cdef arb_ptr psipow = self.tmp + 6
        arb_dot(ell, NULL, 0, self.x, 1, self.vertices + v * self.d, 1, self.d, prec)
        if self.need_log:
            arb_log(logell, ell, prec)
        if self.M == 1:
            # T_v q_0 c_v R(sigma): -log l on the full dimension, 1/l on
            # a hyperplane
            arb_mul(t, self.gamma, self.T + v * self.NS, prec)
            arb_mul(t, t, self.q, prec)
            if self.shift == 0:
                arb_submul(value, t, logell, prec)
            else:
                arb_div(t, t, ell, prec)
                arb_add(value, value, t, prec)
            return
        if self.shift == 1 and self.E == 0:
            # R(i + 1) psi^i/i! = z^i/l, z = psi/l, so that G_v = T_v l^{-1}
            # sum_n q_n x_s^n V_n, V_n = z V_{n-1} + c_{v,0,n}
            arb_inv(inv, ell, prec)
            arb_mul(z, self.psi_values + v, inv, prec)
            arb_set(self.Vn, self.gamma)
            for i in range(1, self.DQ + 1):
                arb_set(self.Vn + i, self.gamma + i)
                arb_addmul(self.Vn + i, self.Vn + i - 1, z, prec)
            arb_dot(total, NULL, 0, self.qx, 1, self.Vn, 1, self.DQ + 1, prec)
            arb_mul(total, total, inv, prec)
            arb_addmul(value, total, self.T + v * self.NS, prec)
            return
        # R(a) at index a + E, for a = -E .. DQ + 1
        # Here we have shift = 0 or E > 0, so need_log is true.
        arb_one(self.ellpow)
        for e in range(1, self.E + 1):
            arb_mul(self.ellpow + e, self.ellpow + e - 1, ell, prec)
        for e in range(self.E + 1):
            # (-1)^e/e! l^e (H_e - log l)
            arb_sub(t, self.hconst_harm + e, logell, prec)
            arb_mul(t, t, self.ellpow + e, prec)
            arb_mul(self.R + self.E - e, t, self.hconst_fact + e, prec)
        arb_one(self.iellpow)
        arb_inv(self.iellpow + 1, ell, prec)
        for i in range(2, self.DQ + 2):
            arb_mul(self.iellpow + i, self.iellpow + i - 1, self.iellpow + 1, prec)
        for i in range(1, self.DQ + 2):
            arb_mul(self.R + self.E + i, self.factorial + i - 1, self.iellpow + i, prec)
        # zk[i, e] = psi(v)^i/i! R(i - e + sigma), then
        # Ssum[l, e] = sum_i q_{l+i} x_s^{l+i} zk[i, e]
        arb_one(psipow)
        for i in range(self.DQ + 1):
            if i > 0:
                arb_mul(psipow, psipow, self.psi_values + v, prec)
                arb_div_si(psipow, psipow, i, prec)
            for e in range(self.E + 1):
                arb_mul(self.zk + i * (self.E + 1) + e, psipow,
                        self.R + self.E + i - e + self.shift, prec)
        for l in range(self.DQ + 1):
            for e in range(self.E + 1):
                arb_dot(self.Ssum + l * (self.E + 1) + e, NULL, 0, self.qx + l, 1,
                        self.zk + e, self.E + 1, self.DQ + 1 - l, prec)
        # u_s = sum_l c_{v,beta_s,l} Ssum[l, |beta_s|], then
        # value += sum_s T_{v,beta_s} u_s
        for i in range(self.NS):
            arb_dot(self.u + i, NULL, 0, self.gamma + i * (self.DQ + 1), 1,
                    self.Ssum + self.degree[i * (self.DQ + 1)], self.E + 1,
                    self.DQ + 1, prec)
        arb_dot(value, value, 0, self.T + v * self.NS, 1, self.u, 1, self.NS, prec)

    cdef void _integrand(self, arb_ptr value):
        # G at the node in self.x
        cdef slong g, f, c, i, idx, p, nf1, v, current
        cdef long *gens
        cdef long *signs
        cdef long *row
        cdef slong prec = self.prec
        cdef arb_ptr t = self.tmp
        cdef arb_ptr pi = self.tmp + 7
        nf1 = self.F + 1
        for g in range(self.ngen):
            arb_dot(self.a + g, NULL, 0, self.x, 1, self.generators + g * self.d, 1,
                    self.d, prec)
            arb_inv(self.ia + g, self.a + g, prec)
            for f in range(self.F):
                arb_mul(self.y + g * self.F + f, self.slopes + g * self.F + f,
                        self.ia + g, prec)
            if self.DQ > 0:
                arb_mul(self.ypsi + g, self.psi_slopes + g, self.ia + g, prec)
        # x_s^n q_n
        arb_set(self.qx, self.q)
        if self.DQ > 0:
            arb_one(t)
            for i in range(1, self.DQ + 1):
                arb_mul(t, t, self.x + self.divisor, prec)
                arb_mul(self.qx + i, self.q + i, t, prec)

        arb_zero(value)
        current = -1
        for c in range(self.ncones):
            v = self.cone_vertex[c]
            if v != current:
                if current >= 0:
                    self._vertex_term(current, value)
                current = v
                for idx in range(self.M):
                    arb_zero(self.gamma + idx)
            arb_set(pi, self.dets + c)
            gens = &self.cone_gens[c * self.rank]
            signs = &self.cone_signs[c * self.rank]
            for i in range(self.rank):
                arb_mul(pi, pi, self.ia + gens[i], prec)
                if signs[i] < 0:
                    arb_neg(pi, pi)
            if self.M == 1:
                arb_add(self.gamma, self.gamma, pi, prec)
                continue
            arb_one(self.H)
            for idx in range(1, self.M):
                arb_zero(self.H + idx)
            for i in range(self.rank):
                g = gens[i]
                # Y^C <- Y^C / (1 - sum_f zeta_f alpha_{f,g}
                #                - xi alpha_{psi,g}), in place
                for idx in range(1, self.M):
                    row = &self.pred[idx * nf1]
                    for f in range(self.F):
                        p = row[f]
                        if p >= 0:
                            arb_addmul(self.H + idx, self.y + g * self.F + f,
                                       self.H + p, prec)
                    p = row[self.F]
                    if p >= 0:
                        arb_addmul(self.H + idx, self.ypsi + g, self.H + p, prec)
            for idx in range(self.M):
                arb_addmul(self.gamma + idx, pi, self.H + idx, prec)
        self._vertex_term(current, value)

    cdef void _point(self, long *index, arb_ptr weight):
        # the node of the multi-index, into self.x, and its weight
        cdef slong i, n
        cdef arb_ptr prefix = self.tmp
        arb_one(prefix)
        arb_one(weight)
        for i in range(self.k):
            n = self.offsets[i] + index[i]
            arb_mul(self.x + self.order[i], prefix, self.nodes + n, self.prec)
            arb_mul(prefix, prefix, self.complements + n, self.prec)
            arb_mul(weight, weight, self.weights + n, self.prec)
        arb_set(self.x + self.order[self.k], prefix)

    cdef bint _accept(self, arb_ptr value):
        if not arb_is_finite(value):
            return False
        return (not self.has_tolerance
                or mag_cmp_2exp_si(arb_radref(value), self.tolerance_exponent) <= 0)

    def run(self, slong first_lo=0, slong first_hi=-1, explicit=None):
        r"""
        Return the sum of `w(x) G(x)` over the nodes whose first index lies
        in ``[first_lo, first_hi)`` (all of them if ``first_hi`` is `-1`), or
        over the multi-indices of ``explicit``, and the list of the
        multi-indices of the nodes left out.

        EXAMPLES:

        The segment `t_1 + t_2 = 1`, `1/4 \leq t_1 \leq 1/2`, on which
        `\int dt_1/(t_1 t_2) = \log 3`; the prefactor `1/c_2` of the
        hyperplane is applied by hand::

            sage: from sieve_integral_brion._brion_sum import BrionSum
            sage: from sieve_integral_brion.geometry import vertex_cones, generator_table
            sage: from sieve_integral_brion.rules import conical_factors
            sage: segment = Polyhedron(vertices=[(1/4, 3/4), (1/2, 1/2)])
            sage: vertices, cones = vertex_cones(segment, (3/8, 5/8), True)
            sage: generators, cones = generator_table(cones)
            sage: brion_sum = BrionSum(80, conical_factors([12], 80), vertices,
            ....:                      generators, cones, 1)
            sage: value, failed = brion_sum.run()
            sage: abs(value / (5/8) - log(RBF(3))) < 10^-10
            True

        The nodes split by their first index::

            sage: first, _ = brion_sum.run(0, 5)
            sage: second, _ = brion_sum.run(5, 12)
            sage: (first + second).overlaps(value)
            True
            sage: brion_sum.run(explicit=[(0,), (1,)])[0].overlaps(brion_sum.run(0, 2)[0])
            True
        """
        cdef RealBall result = RealBall.__new__(RealBall)
        cdef slong i, limit
        cdef bint done
        cdef long *index = <long *> PyMem_Malloc(max(1, self.k) * sizeof(long))
        cdef arb_t weight, value, total
        if index == NULL:
            raise MemoryError
        failed = []
        arb_init(weight)
        arb_init(value)
        arb_init(total)
        try:
            arb_zero(total)
            if explicit is not None:
                for multi in explicit:
                    for i in range(self.k):
                        index[i] = multi[i]
                    self._point(index, weight)
                    self._integrand(value)
                    if self._accept(value):
                        arb_addmul(total, weight, value, self.prec)
                    else:
                        failed.append(tuple(multi))
            else:
                if first_hi < 0 or first_hi > self.msize[0]:
                    first_hi = self.msize[0]
                for i in range(self.k):
                    index[i] = 0
                index[0] = first_lo
                done = first_lo >= first_hi
                while not done:
                    self._point(index, weight)
                    self._integrand(value)
                    if self._accept(value):
                        arb_addmul(total, weight, value, self.prec)
                    else:
                        failed.append(tuple(index[i] for i in range(self.k)))
                    # odometer, the last index fastest
                    i = self.k - 1
                    while True:
                        index[i] += 1
                        limit = first_hi if i == 0 else self.msize[i]
                        if index[i] < limit:
                            break
                        if i == 0:
                            done = True
                            break
                        index[i] = 0
                        i -= 1
            result._parent = RealBallField(self.prec)
            arb_set(result.value, total)
        finally:
            arb_clear(weight)
            arb_clear(value)
            arb_clear(total)
            PyMem_Free(index)
        return result, failed

    def zero_generators(self, multi):
        r"""
        Return the indices of the generators `g` with `\langle x, g\rangle = 0`
        (meaning that the dot product -- an arb ball -- contains `0`),
        for `x` the node at multi-index ``multi``.

        EXAMPLES:

        On the segment of `t_1 + t_2 = 1` from `(1/4, 3/4)` to `(3/4, 1/4)`,
        whose centres are equal, the middle node of three has `x_1 = x_2`::

            sage: from sieve_integral_brion._brion_sum import BrionSum
            sage: from sieve_integral_brion.geometry import vertex_cones, generator_table
            sage: from sieve_integral_brion.rules import conical_factors
            sage: segment = Polyhedron(vertices=[(1/4, 3/4), (3/4, 1/4)])
            sage: vertices, cones = vertex_cones(segment, (1/2, 1/2), True)
            sage: generators, cones = generator_table(cones)
            sage: brion_sum = BrionSum(64, conical_factors([3], 64), vertices,
            ....:                      generators, cones, 1)
            sage: generators, [brion_sum.zero_generators((i,)) for i in range(3)]
            ([(1, -1)], [[], [0], []])
        """
        cdef slong i, g
        cdef long *index
        cdef arb_t weight
        if len(multi) != self.k:
            raise ValueError(f"{multi} is not a multi-index of the rule")
        for i in range(self.k):
            if not 0 <= multi[i] < self.msize[i]:
                raise ValueError(f"{multi} is not a multi-index of the rule")
        index = <long *> PyMem_Malloc(self.k * sizeof(long))
        if index == NULL:
            raise MemoryError
        arb_init(weight)
        zeros = []
        try:
            for i in range(self.k):
                index[i] = multi[i]
            self._point(index, weight)
            for g in range(self.ngen):
                arb_dot(self.a + g, NULL, 0, self.x, 1, self.generators + g * self.d, 1,
                        self.d, self.prec)
                if arb_contains_zero(self.a + g):
                    zeros.append(g)
        finally:
            arb_clear(weight)
            PyMem_Free(index)
        return zeros
