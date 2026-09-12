"""
Text preprocessing for ED-05: canonicalization and boilerplate stripping.
These functions are applied at BOTH train and inference time to ensure
robustness to CF1-CF3 style transforms.
"""

import re


# Boilerplate strings (used by CF2 and CF3)
CF2_PREFIX = "in my answer, i think that"
CF3_SUFFIX = "this is my final answer."


def canonicalize(text: str) -> str:
    """Canonicalize text: lowercase + strip punctuation via [^\\w\\s] removal.

    This is IDEMPOTENT: calling it twice gives the same result as once.
    Uses [^\\w\\s] regex, NOT string.punctuation translation table.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def strip_boilerplate(text: str) -> str:
    """Strip known CF2/CF3 boilerplate strings (case-insensitive).

    CF2 prefix: "In my answer, I think that "
    CF3 suffix: " This is my final answer."
    """
    # Strip CF2 prefix (case-insensitive)
    lower = text.lower()
    # Try with trailing space first, then without
    for prefix in [CF2_PREFIX + " ", CF2_PREFIX]:
        if lower.startswith(prefix):
            text = text[len(prefix):]
            break

    # Strip CF3 suffix (case-insensitive)
    lower = text.lower()
    for suffix in [" " + CF3_SUFFIX, CF3_SUFFIX]:
        if lower.endswith(suffix):
            text = text[:-len(suffix)]
            break

    return text.strip()


def preprocess(text: str) -> str:
    """Full preprocessing pipeline: strip boilerplate, then canonicalize.

    This is the SINGLE ENTRY POINT used everywhere in the pipeline.
    """
    text = strip_boilerplate(text)
    text = canonicalize(text)
    return text


if __name__ == '__main__':
    print("=" * 60)
    print("  Preprocessing Tests")
    print("=" * 60)

    # Test canonicalize
    tests_canon = [
        ("Don't touch!", "dont touch"),
        ("single-pole switch", "singlepole switch"),
        ("Hello, World!!!", "hello world"),
        ("  Multiple   spaces  ", "multiple spaces"),
        ("already clean", "already clean"),
    ]
    print("\n--- canonicalize() ---")
    for inp, expected in tests_canon:
        result = canonicalize(inp)
        # Check idempotence
        result2 = canonicalize(result)
        status = "PASS" if result == expected and result2 == result else "FAIL"
        print(f"  {status}: canonicalize({inp!r}) = {result!r} (expected {expected!r}, idempotent={result2==result})")

    # Test strip_boilerplate
    tests_bp = [
        ("In my answer, I think that the water evaporated", "the water evaporated"),
        ("the water evaporated This is my final answer.", "the water evaporated"),
        ("In my answer, I think that the water evaporated This is my final answer.", "the water evaporated"),
        ("Normal answer without boilerplate", "Normal answer without boilerplate"),
    ]
    print("\n--- strip_boilerplate() ---")
    for inp, expected in tests_bp:
        result = strip_boilerplate(inp)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: strip_boilerplate({inp!r})")
        print(f"         = {result!r} (expected {expected!r})")

    # Test preprocess (combined)
    tests_preprocess = [
        ("In my answer, I think that Don't touch!", "dont touch"),
        ("The answer is YES! This is my final answer.", "the answer is yes"),
        ("In my answer, I think that it's simple. This is my final answer.", "its simple"),
    ]
    print("\n--- preprocess() ---")
    for inp, expected in tests_preprocess:
        result = preprocess(inp)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: preprocess({inp!r})")
        print(f"         = {result!r} (expected {expected!r})")
