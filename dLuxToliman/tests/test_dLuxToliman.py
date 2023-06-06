def test_module_import():
    try:
        import dLuxToliman
    except ImportError:
        raise ImportError("Failed to import dLuxToliman")
    return
