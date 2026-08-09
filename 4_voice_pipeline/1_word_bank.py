"""
Hand-curated, fixed-content sentences for the af_heart training corpus.

Everything in here is verified programmatically by
qa/test_word_bank.py (not just asserted by eye) — e.g. every entry in
PANGRAMS is checked to actually contain all 26 letters before it is
trusted, rather than assuming a remembered pangram is correct.
"""

# Each of these is claimed to contain every letter a-z at least once.
# qa/test_word_bank.py verifies this claim; do not add an entry here
# without also being able to pass that check.
PANGRAMS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Waltz, bad nymph, for quick jigs vex.",
    "Jinxed wizards pluck ivy from the big quilt.",
    "Crazy Fredrick bought many very exquisite opal jewels.",
    "We promptly judged antique ivory buckles for the next prize.",
    "A quick movement of the enemy will jeopardize six gunboats.",
]

# Cardinal numbers, spelled out, so the model sees written-number
# pronunciation across a wide numeric range.
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _number_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rem = divmod(n, 10)
        word = _TENS[tens - 2]
        return f"{word}-{_ONES[rem]}" if rem else word
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        word = f"{_ONES[hundreds]} hundred"
        return f"{word} and {_number_to_words(rem)}" if rem else word
    thousands, rem = divmod(n, 1000)
    word = f"{_number_to_words(thousands)} thousand"
    return f"{word} {_number_to_words(rem)}" if rem else word


NUMBER_SENTENCES = [
    f"I counted {_number_to_words(n)} apples on the table."
    for n in (0, 1, 2, 7, 12, 15, 19, 20, 21, 35, 48, 59, 60, 77, 89, 90,
              99, 100, 150, 365, 999, 1000, 2026, 8675)
] + [
    "The train departs at half past three in the afternoon.",
    "Please call me back at a quarter to nine tonight.",
    "The total comes to twelve dollars and fifty cents.",
    "My meeting starts in exactly forty-five minutes.",
    "She was born on the twenty-first of March.",
    "The recipe needs two and a half cups of flour.",
    "Add three point one four to the running total.",
    "The score was seven to two at halftime.",
]

# Common contractions, deliberately packed in so the G2P layer sees
# them in natural context rather than only in isolation.
CONTRACTION_SENTENCES = [
    "I don't think we'll make it in time.",
    "She isn't coming, but they're on their way.",
    "You shouldn't've eaten that whole cake by yourself.",
    "We're excited, but I'm a little nervous too.",
    "It's not what it looks like, I promise.",
    "They've already left, and we haven't caught up yet.",
    "Wouldn't it be nice if it didn't rain today?",
    "He'd rather stay home than go to the party.",
    "Let's not forget what we've learned here.",
    "Can't you see that it's already working?",
]

# Varied sentence forms: statements, questions, exclamations, lists,
# quotes, parentheticals, abbreviations — punctuation diversity the
# LLM-generated batch may not reliably hit on its own.
PUNCTUATION_SENTENCES = [
    "Is this really the fastest way to get there?",
    "Watch out! The floor is still wet.",
    "We need eggs, milk, bread, and a dozen apples.",
    "Dr. Alvarez said the results looked fine.",
    "My favorite color is blue, not green.",
    "\"I'll be right there,\" she said, grabbing her coat.",
    "The meeting (originally set for Monday) moved to Thursday.",
    "Wait... did you hear that noise?",
    "Mr. Chen runs the shop on Fifth and Main.",
    "Well, that's one way to solve it, I suppose.",
]


def fixed_sentences() -> list[str]:
    """All hand-curated sentences, source-tagged, deduplicated in build_corpus.py."""
    return (
        PANGRAMS
        + NUMBER_SENTENCES
        + CONTRACTION_SENTENCES
        + PUNCTUATION_SENTENCES
    )
