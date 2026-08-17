"""kerze.testing — test utilities for kerze itself (gradcheck, etc).
Not part of the public modeling API; imported by kerze's own test suite
and available to users who want to gradcheck their own custom ops."""

from kerze.testing.gradcheck import gradcheck, numerical_gradient

__all__ = ["gradcheck", "numerical_gradient"]
