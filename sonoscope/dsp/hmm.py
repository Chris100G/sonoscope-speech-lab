"""Discrete hidden Markov models and Baum-Welch training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate(
    observations: np.ndarray, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(observations, dtype=int).reshape(-1)
    pi, trans, emit = (
        np.asarray(x, dtype=float) for x in (initial, transition, emission)
    )
    states = pi.size
    if obs.size == 0 or states == 0:
        raise ValueError("observations and states must be non-empty")
    if trans.shape != (states, states) or emit.ndim != 2 or emit.shape[0] != states:
        raise ValueError("incompatible HMM parameter shapes")
    if np.any(obs < 0) or np.any(obs >= emit.shape[1]):
        raise ValueError("observation symbol outside emission table")
    if np.any(pi < 0) or np.any(trans < 0) or np.any(emit < 0):
        raise ValueError("probabilities cannot be negative")
    return obs, pi, trans, emit


def _scaled_forward(obs: np.ndarray, pi: np.ndarray, trans: np.ndarray, emit: np.ndarray):
    states, length = len(pi), len(obs)
    alpha = np.zeros((length, states))
    scales = np.zeros(length)
    alpha[0] = pi * emit[:, obs[0]]
    scales[0] = max(alpha[0].sum(), np.finfo(float).tiny)
    alpha[0] /= scales[0]
    for t in range(1, length):
        alpha[t] = (alpha[t - 1] @ trans) * emit[:, obs[t]]
        scales[t] = max(alpha[t].sum(), np.finfo(float).tiny)
        alpha[t] /= scales[t]
    return float(np.log(scales).sum()), alpha, scales


def forward(
    observations: np.ndarray, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return log-likelihood and per-frame normalized forward probabilities."""
    obs, pi, trans, emit = _validate(observations, initial, transition, emission)
    score, alpha, _ = _scaled_forward(obs, pi, trans, emit)
    return score, alpha


def backward(
    observations: np.ndarray, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return log-likelihood and scaled backward probabilities."""
    obs, pi, trans, emit = _validate(observations, initial, transition, emission)
    score, _, scales = _scaled_forward(obs, pi, trans, emit)
    beta = np.ones((len(obs), len(pi)))
    for t in range(len(obs) - 2, -1, -1):
        beta[t] = trans @ (emit[:, obs[t + 1]] * beta[t + 1]) / scales[t + 1]
    return score, beta


def viterbi(
    observations: np.ndarray, initial: np.ndarray, transition: np.ndarray, emission: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the most likely state path and its log probability."""
    obs, pi, trans, emit = _validate(observations, initial, transition, emission)
    tiny = np.finfo(float).tiny
    log_pi, log_trans, log_emit = np.log(np.maximum(pi, tiny)), np.log(np.maximum(trans, tiny)), np.log(np.maximum(emit, tiny))
    score = log_pi + log_emit[:, obs[0]]
    back = np.zeros((len(obs), len(pi)), dtype=int)
    for t in range(1, len(obs)):
        candidates = score[:, None] + log_trans
        back[t] = np.argmax(candidates, axis=0)
        score = np.max(candidates, axis=0) + log_emit[:, obs[t]]
    state = int(np.argmax(score))
    path = np.empty(len(obs), dtype=int)
    path[-1] = state
    for t in range(len(obs) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return float(np.max(score)), path


def baum_welch(
    sequences: list[np.ndarray], n_states: int, n_symbols: int,
    iterations: int = 20, smoothing: float = 1e-6, seed: int = 0,
) -> DiscreteHMM:
    """Train a discrete HMM with scaled Baum-Welch expectation maximization."""
    if not sequences or n_states <= 0 or n_symbols <= 0 or iterations <= 0:
        raise ValueError("sequences, n_states, n_symbols, and iterations must be positive")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    seqs = [np.asarray(seq, dtype=int).reshape(-1) for seq in sequences]
    if any(s.size == 0 or np.any(s < 0) or np.any(s >= n_symbols) for s in seqs):
        raise ValueError("sequences must contain valid non-empty symbols")
    rng = np.random.default_rng(seed)
    pi = np.full(n_states, 1 / n_states)
    trans = rng.random((n_states, n_states)) + smoothing
    trans /= trans.sum(axis=1, keepdims=True)
    emit = rng.random((n_states, n_symbols)) + smoothing
    emit /= emit.sum(axis=1, keepdims=True)
    for _ in range(iterations):
        pi_count = np.full(n_states, smoothing)
        trans_count = np.full((n_states, n_states), smoothing)
        emit_count = np.full((n_states, n_symbols), smoothing)
        for obs in seqs:
            _, alpha, scales = _scaled_forward(obs, pi, trans, emit)
            beta = np.ones_like(alpha)
            for t in range(len(obs) - 2, -1, -1):
                beta[t] = trans @ (emit[:, obs[t + 1]] * beta[t + 1]) / scales[t + 1]
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), np.finfo(float).tiny)
            pi_count += gamma[0]
            for t, symbol in enumerate(obs):
                emit_count[:, symbol] += gamma[t]
            for t in range(len(obs) - 1):
                xi = alpha[t, :, None] * trans * (emit[:, obs[t + 1]] * beta[t + 1])[None, :]
                xi /= max(xi.sum(), np.finfo(float).tiny)
                trans_count += xi
        pi = pi_count / pi_count.sum()
        trans = trans_count / trans_count.sum(axis=1, keepdims=True)
        emit = emit_count / emit_count.sum(axis=1, keepdims=True)
    return DiscreteHMM(pi, trans, emit)


@dataclass
class DiscreteHMM:
    """A discrete-symbol HMM with convenience inference methods."""

    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray

    def score(self, observations: np.ndarray) -> float:
        return forward(observations, self.initial, self.transition, self.emission)[0]

    def forward(self, observations: np.ndarray) -> tuple[float, np.ndarray]:
        return forward(observations, self.initial, self.transition, self.emission)

    def backward(self, observations: np.ndarray) -> tuple[float, np.ndarray]:
        return backward(observations, self.initial, self.transition, self.emission)

    def viterbi(self, observations: np.ndarray) -> tuple[float, np.ndarray]:
        return viterbi(observations, self.initial, self.transition, self.emission)
