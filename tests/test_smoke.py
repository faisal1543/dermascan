"""Smoke test so CI is green from day one. Replace with real tests as code lands."""


def test_packages_import():
    import app  # noqa: F401
    import perception  # noqa: F401
    import reasoning  # noqa: F401
