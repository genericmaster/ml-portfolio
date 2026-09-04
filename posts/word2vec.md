---
title: "Word2Vec from scratch: two implementations, one corpus"
date: 2026-08-28
tag: FROM SCRATCH
---

# Word2Vec from scratch: two implementations, one corpus

This is a placeholder post. Replace with your actual writeup.

## The math

The probability of context word $o$ given center word $c$ is:

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

## What I built

Two implementations on the Harry Potter corpus...
