from decimal import Decimal


def fp(x):
    """Human price: 64321.5 -> '64321.5', 0.000004123 -> '0.000004123' (never '0.00' or '4e-06')."""
    if x is None:
        return "-"
    return f"{Decimal(f'{float(x):.8g}'):f}"
