# Shadow-rate VAR

## Motivation

A shadow-rate VAR is a VAR framework designed for settings where an observed policy rate is constrained by an **effective lower bound (ELB)**.

The key idea is to distinguish between:
- an observed short rate $i_t$ (censored at the ELB), and
- a latent *shadow* rate $s_t$ that can move below the ELB.

A common measurement (censoring) relationship is:

$$
 i_t = \max\{\mathrm{ELB},\, s_t\}.
$$

## Block-hybrid intuition (paper background)

The paper motivating this toolkit discusses *block-hybrid* variants in which:
- macroeconomic variables respond to lagged **observed** rates, whilst
- financial variables may load on lagged **shadow** rates.

This is intended to capture a distinction between:
- economic agents facing administered rates, and
- financial markets pricing off shadow policy expectations.

## In this toolkit

The Python toolkit implements a practical SRVAR workflow centred on:
- reduced-form Bayesian VARs,
- ELB/shadow-rate **data augmentation** (sampling latent values at the bound), and
- optional diagonal stochastic volatility.

It does **not** currently implement the full structural SVAR/block-hybrid state-space system described in the paper. Instead, it provides an ELB-aware VAR likelihood by treating ELB-bound observations as latent and sampling them subject to the constraint.

Next pages:
- {doc}`theory/elb`
- {doc}`theory/mcmc`
