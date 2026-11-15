"""Version 1: 1,024 digits, including the integer 3, without punctuation.

Integer Machin arctangent series; independently checked with Decimal Chudnovsky
in tests. Ordinary binary floating point is never used to generate the digits.
"""
from pathlib import Path

PI_DIGITS = Path(__file__).with_name("pi_digits.txt").read_text().strip()
if len(PI_DIGITS) != 1024 or not PI_DIGITS.isascii() or not PI_DIGITS.isdigit():
    raise ValueError("invalid Pi Fly v1 reference")


def machin_digits(count=1024):
    """Offline generation/verification; guard digits absorb series rounding."""
    scale = 10 ** (count + 30)

    def atan_inverse(q):
        power, total, k, sign = scale // q, 0, 0, 1
        while power:
            total += sign * (power // (2 * k + 1))
            power //= q * q
            k += 1
            sign = -sign
        return total

    return str(16 * atan_inverse(5) - 4 * atan_inverse(239))[:count]
