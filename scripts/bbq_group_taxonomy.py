#!/usr/bin/env python3
"""Canonical group taxonomy for the BBQ-derived bias datasets (single source of truth).

For each BBQ category this defines, EXPLICITLY and editably:
  - `map`:       BBQ `Known_stereotyped_groups` token  ->  canonical group
  - `surface`:   canonical group  ->  (copular_form, groupnoun_form)
                 copular_form  goes in  "{Name} is {copular} and {predicate}"   (article baked in)
                 groupnoun_form goes in "{groupnoun} are {predicate}"
  - `reference`: canonical group  ->  the dominant/contrast group it is paired against
                 (a dict; categories with one dominant map everything to it)
  - `axis`:      a coarse grouping label for later segmentation

Judgment calls (made visible, not hidden): Age & Nationality have no natural "dominant" — we pair
old↔young and each nationality↔American respectively. Physical_appearance & Gender use per-target
references (obese↔thin, short↔tall; woman↔man, trans↔cis). Edit freely.
"""
from __future__ import annotations

# Each category: dict(axis, dominant_default, map{token->canon}, surface{canon->(copular, groupnoun)},
#                      reference{canon->canon}  (optional; falls back to dominant_default))
CATEGORIES: dict[str, dict] = {
    "Race_ethnicity": {
        "axis": "race",
        "dominant_default": "White",
        "map": {"Black": "Black", "African American": "Black", "Asian": "Asian",
                "Hispanic": "Hispanic", "Latino": "Hispanic", "Middle Eastern": "Middle Eastern",
                "Arab": "Middle Eastern", "Native American": "Native American", "Jewish": "Jewish",
                "Roma": "Roma", "White": "White"},
        "surface": {"Black": ("Black", "Black people"), "Asian": ("Asian", "Asian people"),
                    "Hispanic": ("Hispanic", "Hispanic people"),
                    "Middle Eastern": ("Middle Eastern", "Middle Eastern people"),
                    "Native American": ("Native American", "Native American people"),
                    "Jewish": ("Jewish", "Jewish people"), "Roma": ("Roma", "Roma people"),
                    "White": ("White", "White people")},
    },
    "Religion": {
        "axis": "religion",
        "dominant_default": "Christian",
        "map": {"Muslim": "Muslim", "Jewish": "Jewish", "Hindu": "Hindu", "Mormon": "Mormon",
                "Atheist": "Atheist", "Orthodox": "Orthodox Jewish", "Catholic": "Catholic",
                "Christian": "Christian"},
        "surface": {"Muslim": ("Muslim", "Muslims"), "Jewish": ("Jewish", "Jewish people"),
                    "Hindu": ("Hindu", "Hindus"), "Mormon": ("Mormon", "Mormons"),
                    "Atheist": ("an atheist", "atheists"), "Orthodox Jewish": ("Orthodox Jewish", "Orthodox Jews"),
                    "Catholic": ("Catholic", "Catholics"), "Christian": ("Christian", "Christians")},
    },
    "SES": {
        "axis": "socioeconomic",
        "dominant_default": "rich",
        "map": {"low SES": "poor", "lowSES": "poor", "high SES": "rich", "highSES": "rich"},
        "surface": {"poor": ("poor", "poor people"), "rich": ("rich", "rich people")},
    },
    "Sexual_orientation": {
        "axis": "sexual_orientation",
        "dominant_default": "straight",
        "map": {"gay": "gay", "lesbian": "lesbian", "bisexual": "bisexual", "pansexual": "pansexual",
                "straight": "straight"},
        "surface": {"gay": ("gay", "gay people"), "lesbian": ("a lesbian", "lesbians"),
                    "bisexual": ("bisexual", "bisexual people"), "pansexual": ("pansexual", "pansexual people"),
                    "straight": ("straight", "straight people")},
    },
    "Disability_status": {
        "axis": "disability",
        "dominant_default": "nondisabled",
        "map": {"disabled": "disabled", "physically disabled": "physically disabled",
                "mentally-ill": "mentally ill", "D/deaf": "deaf",
                "autistic people": "autistic", "people with blindness or low-vision": "blind",
                "people with cerebral palsy": "disabled", "Down's syndrome": "disabled",
                "people with cognitive disabilities or mental illness": "mentally ill",
                "nondisabled": "nondisabled"},
        "surface": {"disabled": ("disabled", "disabled people"),
                    "physically disabled": ("physically disabled", "physically disabled people"),
                    "mentally ill": ("mentally ill", "mentally ill people"),
                    "deaf": ("deaf", "deaf people"), "autistic": ("autistic", "autistic people"),
                    "blind": ("blind", "blind people"), "nondisabled": ("able-bodied", "able-bodied people")},
    },
    "Physical_appearance": {
        "axis": "physical_appearance",
        "dominant_default": "average-looking",
        "map": {"obese": "obese", "short": "short", "pregnant": "pregnant",
                "negDress": "badly dressed", "visibleDifference": "physically scarred"},
        "surface": {"obese": ("obese", "obese people"), "short": ("short", "short people"),
                    "pregnant": ("pregnant", "pregnant people"),
                    "badly dressed": ("badly dressed", "badly dressed people"),
                    "physically scarred": ("physically scarred", "physically scarred people"),
                    "thin": ("thin", "thin people"), "tall": ("tall", "tall people"),
                    "not pregnant": ("not pregnant", "people who are not pregnant"),
                    "well dressed": ("well dressed", "well dressed people"),
                    "average-looking": ("average-looking", "average-looking people")},
        "reference": {"obese": "thin", "short": "tall", "pregnant": "not pregnant",
                      "badly dressed": "well dressed", "physically scarred": "average-looking"},
    },
    "Age": {
        "axis": "age",
        "dominant_default": "young",
        "map": {"old": "old", "nonOld": "young"},
        "surface": {"old": ("old", "old people"), "young": ("young", "young people")},
        "reference": {"old": "young", "young": "old"},
    },
    "Gender_identity": {
        "axis": "gender",
        "dominant_default": "a man",
        "map": {"F": "a woman", "M": "a man", "Transgender women": "a transgender woman",
                "transgender women": "a transgender woman", "transgender men": "a transgender man",
                "trans": "transgender"},
        "surface": {"a woman": ("a woman", "women"), "a man": ("a man", "men"),
                    "a transgender woman": ("a transgender woman", "transgender women"),
                    "a transgender man": ("a transgender man", "transgender men"),
                    "transgender": ("transgender", "transgender people"),
                    "a cisgender woman": ("a cisgender woman", "cisgender women"),
                    "a cisgender man": ("a cisgender man", "cisgender men"),
                    "cisgender": ("cisgender", "cisgender people")},
        "reference": {"a woman": "a man", "a man": "a woman",
                      "a transgender woman": "a cisgender woman", "a transgender man": "a cisgender man",
                      "transgender": "cisgender"},
    },
    "Nationality": {
        "axis": "nationality",
        "dominant_default": "American",
        # identity map: nationalities are used as-is (adjectival). American is the reference.
        "map": {n: n for n in [
            "Afghan", "British", "Burmese", "Chinese", "Eritrean", "Ethiopian", "Guinean", "Indian",
            "Indonesian", "Iranian", "Iraqi", "Irish", "Italian", "Japanese", "Kenyan", "Korean",
            "Libyan", "Malian", "Moroccan", "Mozambican", "Namibian", "Nigerian", "Pakistani",
            "Palestinian", "Saudi", "Sri Lankan", "Syrian", "Thai", "Vietnamese", "Yemeni", "American"]},
        "surface": {n: (n, f"{n} people") for n in [
            "Afghan", "British", "Burmese", "Chinese", "Eritrean", "Ethiopian", "Guinean", "Indian",
            "Indonesian", "Iranian", "Iraqi", "Irish", "Italian", "Japanese", "Kenyan", "Korean",
            "Libyan", "Malian", "Moroccan", "Mozambican", "Namibian", "Nigerian", "Pakistani",
            "Palestinian", "Saudi", "Sri Lankan", "Syrian", "Thai", "Vietnamese", "Yemeni", "American"]},
    },
}

# Neutral, group-agnostic first names for the name-based frame (avoid race/gender-coded names).
NEUTRAL_NAMES = ["Alex", "Casey", "Jordan", "Taylor", "Riley", "Quinn", "Morgan", "Avery", "Sam", "Jamie"]


def canonicalize(category: str, token: str) -> str | None:
    cat = CATEGORIES.get(category)
    if not cat:
        return None
    return cat["map"].get(token.strip())


def reference_of(category: str, canon: str) -> str | None:
    cat = CATEGORIES.get(category)
    if not cat:
        return None
    ref = cat.get("reference", {}).get(canon)
    return ref if ref is not None else cat["dominant_default"]


def surface(category: str, canon: str) -> tuple[str, str] | None:
    cat = CATEGORIES.get(category)
    if not cat:
        return None
    return cat["surface"].get(canon)


def axis_of(category: str) -> str | None:
    cat = CATEGORIES.get(category)
    return cat["axis"] if cat else None


def is_supported(category: str) -> bool:
    return category in CATEGORIES
