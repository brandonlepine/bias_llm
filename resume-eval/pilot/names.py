"""Name pool for bootstrap replicates.

Each bootstrap *pair* (queer + control) shares ONE identity (same name, email,
phone, city) so that everything within the pair is byte-identical except the
queer-coded signal span. Across pairs we vary the identity to get statistical
power and to let us test whether the queer-signal effect interacts with the
candidate's perceived gender.

Demographic coding
------------------
- gender: we deliberately VARY perceived gender (male / female / ambiguous) and
  record it as a covariate so a later analysis can estimate a queer x gender
  interaction.
- race/ethnicity: held approximately constant. The first/last names below are
  common, predominantly White-coded U.S. names. This is a *limitation*, not a
  control we can perfectly enforce from names alone; it keeps race from being a
  systematic confound in this first pilot. Swap in a different pool to study
  race x queer interactions later.

Sampling is deterministic given a seed (random.Random), so runs are reproducible.
"""

import random

# (first_name, perceived_gender)
FIRST_NAMES = [
    ("Michael", "male"), ("James", "male"), ("Robert", "male"),
    ("Daniel", "male"), ("Matthew", "male"), ("Andrew", "male"),
    ("Thomas", "male"), ("Joseph", "male"), ("Brian", "male"), ("Kevin", "male"),
    ("Emily", "female"), ("Sarah", "female"), ("Jessica", "female"),
    ("Rachel", "female"), ("Laura", "female"), ("Megan", "female"),
    ("Hannah", "female"), ("Katherine", "female"), ("Allison", "female"),
    ("Claire", "female"),
    ("Taylor", "ambiguous"), ("Jordan", "ambiguous"), ("Casey", "ambiguous"),
    ("Morgan", "ambiguous"), ("Riley", "ambiguous"), ("Quinn", "ambiguous"),
    ("Avery", "ambiguous"), ("Reese", "ambiguous"), ("Skyler", "ambiguous"),
    ("Cameron", "ambiguous"),
]

LAST_NAMES = [
    "Carter", "Hughes", "Bennett", "Foster", "Reynolds", "Coleman", "Sullivan",
    "Brooks", "Powell", "Russell", "Griffin", "Hayes", "Porter", "Wagner",
    "Barrett", "Fletcher", "Hunter", "Mason", "Crawford", "Lawson", "Spencer",
    "Webb", "Chapman", "Walsh", "Donovan", "Pierce", "Holloway", "Whitfield",
    "Sherman", "Banks",
]

CITY = "Goleta, California"
AREA_CODE = "805"


def _identity(rng):
    first, gender = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    full = f"{first} {last}"
    handle = f"{first}{last}".lower()
    email = f"{first}.{last}@email.com".lower()
    # 4-digit local part, identical within a pair, varies across pairs.
    phone = f"({AREA_CODE}) 555-{rng.randint(1000, 9999)}"
    return {
        "first": first,
        "last": last,
        "full": full,
        "gender": gender,
        "email": email,
        "linkedin": handle,
        "phone": phone,
        "city": CITY,
    }


def sample_identities(n, seed=0):
    """Return n unique identities (by full name) deterministically from `seed`."""
    rng = random.Random(seed)
    out, seen, guard = [], set(), 0
    max_unique = len(FIRST_NAMES) * len(LAST_NAMES)
    if n > max_unique:
        raise ValueError(f"requested {n} unique names but pool only supports {max_unique}")
    while len(out) < n:
        ident = _identity(rng)
        guard += 1
        if ident["full"] in seen:
            if guard > 100 * n + 1000:
                raise RuntimeError("could not draw enough unique names; enlarge the pool")
            continue
        seen.add(ident["full"])
        out.append(ident)
    return out
