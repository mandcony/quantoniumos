# test_patch_layer.py
# Minimal pytest shim so pytest-style tests run under plain Python/unittest.
# NOTE: This is a compatibility layer — not full pytest. It keeps behavior sane
# (e.g., raises() truly asserts) without generating extra parametrized cases.

import sys, types, unittest, warnings, importlib

class _Raises:
    def __init__(self, expected):
        self.expected = expected
        self._exc = None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"Did not raise {self.expected}")
        try:
            exp = tuple(self.expected) if isinstance(self.expected, (list, tuple)) else (self.expected,)
            return any(issubclass(exc_type, e) for e in exp)
        except TypeError:
            return issubclass(exc_type, self.expected)

def _identity_decorator(func):  # used by many marks/fixtures
    return func

def _fixture_decorator(*a, **k):
    # Our simple fixture decorator just returns the function unchanged
    def decorator(func):
        return func
    return decorator

def _skip_decorator(reason=None):
    def deco(func):
        def wrapper(*a, **k):
            raise unittest.SkipTest(reason or "skipped by mark.skip")
        wrapper.__name__ = func.__name__
        return wrapper
    if callable(reason):  # Handle @pytest.mark.skip without parentheses
        func, reason = reason, None
        return deco(func)
    return deco

def _skipif_decorator(condition, reason=None):
    def deco(func):
        if condition:
            return _skip_decorator(reason)(func)
        return func
    return deco

def _xfail_decorator(reason=None):
    # Keep execution but do not fail the suite on AssertionError
    def deco(func):
        def wrapper(*a, **k):
            try:
                return func(*a, **k)
            except AssertionError:
                # mimic expected failure
                warnings.warn(f"xfail: {reason or ''}")
        return wrapper
    return deco

def _parametrize_decorator(argnames, argvalues):
    # Collection-time expansion is a pytest feature. We can't replicate cleanly.
    # We mark the function so a custom runner *could* expand, otherwise run once.
    def deco(func):
        setattr(func, "__parametrize__", (argnames, argvalues))
        return func
    return deco

class _MarkObj:
    def __init__(self):
        self.skip = _skip_decorator
        self.skipif = _skipif_decorator
        self.xfail = _xfail_decorator
        self.parametrize = _parametrize_decorator
        
    def __getattr__(self, name):
        # unknown marks become no-ops
        return lambda *a, **k: _identity_decorator

class _PyTestShim(types.ModuleType):
    # public API we emulate
    def __init__(self):
        super().__init__("pytest")
        self.mark = _MarkObj()

    def raises(self, expected):            return _Raises(expected)
    def fixture(self, *a, **k):            return _fixture_decorator(*a, **k)
    def approx(self, x, rel=None, abs=None): return x
    def skip(self, reason=""):             raise unittest.SkipTest(reason)

    def importorskip(self, modname, minversion=None):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            raise unittest.SkipTest(f"skipped: cannot import {modname}")
        return mod

# install into sys.modules so `import pytest` picks up our shim
if "pytest" not in sys.modules:
    sys.modules["pytest"] = _PyTestShim()
