"""Correctness tests for the conditional MAF companion (run before any training)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # tiny; keep off the GPUs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from vmim_maf_companion import ConditionalMAF, conditional_maf_log_prob

D, C = 6, 10
LOG2PI = float(np.log(2.0 * np.pi))


def _transforms(n_transforms=8, hidden=(256, 256)):
    lp = hk.without_apply_rng(hk.transform(
        lambda th, y: conditional_maf_log_prob(th, y, D, C, n_transforms, hidden)))
    tb = hk.without_apply_rng(hk.transform(
        lambda th, y: ConditionalMAF(D, C, n_transforms, hidden).to_base(th, y)[0]))
    return lp, tb


def test_shapes_finite_and_identity_init():
    lp, _ = _transforms()
    key = jax.random.PRNGKey(0)
    th0 = jnp.zeros((1, D)); y0 = jnp.zeros((1, C))
    params = lp.init(key, th0, y0)
    B = 32
    th = jax.random.normal(jax.random.PRNGKey(1), (B, D))
    y = jax.random.normal(jax.random.PRNGKey(2), (B, C))
    out = lp.apply(params, th, y)
    assert out.shape == (B,), out.shape
    assert bool(jnp.all(jnp.isfinite(out)))
    # identity init (zero output layer) -> exactly standard normal log-prob
    base = -0.5 * jnp.sum(th ** 2 + LOG2PI, axis=-1)
    err = float(jnp.max(jnp.abs(out - base)))
    assert err < 1e-5, f"identity-init log_prob != N(0,I): max err {err}"
    print(f"[ok] shapes (B,), finite, identity-init==N(0,I) (max err {err:.2e})")


def test_logdet_matches_autograd_jacobian():
    n_t, hid = 5, (64, 64)
    lp, tb = _transforms(n_t, hid)
    key = jax.random.PRNGKey(3)
    th0 = jnp.zeros((1, D)); y0 = jnp.zeros((1, C))
    params = lp.init(key, th0, y0)
    # perturb ALL params so the flow is non-trivial (not identity). Keep the
    # perturbation modest: a large one makes the *composed* Jacobian wildly
    # ill-conditioned and slogdet loses precision (that's a numpy artefact, not
    # a flow bug — verified separately at scale 0.3 -> cond(J)~1e12).
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(7), len(leaves))
    leaves = [l + 0.05 * jax.random.normal(k, l.shape) for l, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(treedef, leaves)

    th = jax.random.normal(jax.random.PRNGKey(11), (D,))
    y = jax.random.normal(jax.random.PRNGKey(12), (C,))

    def to_base_single(t):
        return tb.apply(params, t[None, :], y[None, :])[0]  # (D,)

    J = jax.jacfwd(to_base_single)(th)              # (D, D)
    logdet_jax = jnp.linalg.slogdet(J)[1]            # log|det J|
    z = to_base_single(th)
    logp = lp.apply(params, th[None, :], y[None, :])[0]
    base = -0.5 * jnp.sum(z ** 2 + LOG2PI)
    logdet_module = logp - base
    err = float(jnp.abs(logdet_module - logdet_jax))
    assert err < 1e-4, f"log-det mismatch vs autograd Jacobian: {err}"
    print(f"[ok] log-det == autograd Jacobian (err {err:.2e}); also a valid density")


def test_overfit_conditional_gaussian():
    """theta|y ~ N(W y, 0.2): a trained MAF should sharply lower NLL."""
    import optax
    lp, _ = _transforms(n_transforms=6, hidden=(128, 128))
    rng = np.random.default_rng(0)
    W = rng.normal(size=(D, C)) * 0.5

    def batch(n, k):
        y = rng.normal(size=(n, C)).astype(np.float32)
        th = (y @ W.T + 0.2 * rng.normal(size=(n, D))).astype(np.float32)
        return jnp.asarray(th), jnp.asarray(y)

    params = lp.init(jax.random.PRNGKey(0), *batch(8, 0))
    opt = optax.adam(2e-3); opt_state = opt.init(params)

    def loss(p, th, y):
        return -jnp.mean(lp.apply(p, th, y))

    @jax.jit
    def step(p, os, th, y):
        l, g = jax.value_and_grad(loss)(p, th, y)
        upd, os = opt.update(g, os, p)
        return optax.apply_updates(p, upd), os, l

    th0, y0 = batch(512, 1)
    nll0 = float(loss(params, th0, y0))
    for i in range(400):
        th, y = batch(512, i + 2)
        params, opt_state, _ = step(params, opt_state, th, y)
    nll1 = float(loss(params, th0, y0))
    assert nll1 < nll0 - 1.0, f"MAF failed to learn: NLL {nll0:.3f} -> {nll1:.3f}"
    print(f"[ok] overfit conditional Gaussian: NLL {nll0:.2f} -> {nll1:.2f}")


if __name__ == "__main__":
    test_shapes_finite_and_identity_init()
    test_logdet_matches_autograd_jacobian()
    test_overfit_conditional_gaussian()
    print("ALL MAF COMPANION TESTS PASSED")
