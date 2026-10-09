"""Deterministically build ``movies_sample.csv``, the SYNTHETIC test fixture (see ``README.md`` in this folder).

Every film, person, place and plot is invented. Titles may accidentally resemble real films and names may coincide
with real people; the Wiki Page column deliberately points at ``..._(synthetic_film)`` URLs that do not exist so that
nobody mistakes a fixture row for a real Wikipedia article. The output follows the exact Kaggle schema of
``jrobischon/wikipedia-movie-plots``.

Design (the fixture backs the retrieval evaluation, so every film needs a premise of its own):

* 36 hand-written anchor films (``story_data/anchors_*.yaml``), 13 of them long (over 250 words) so the chunker
  has work.
* 264 generated films. Each takes ONE story motif (``story_data/motifs_*.yaml``: protagonist situation, central
  object and central twist) drawn WITHOUT replacement, so no two films share a premise. Only connective beats and
  closing sentences (``story_data/shared.yaml``) may repeat, and the beats embed the film's own central object.
* The generated films include 8 stub plots under 50 words (cleaning must drop them) plus one plot of exactly 49 words
  (dropped) and one of exactly 50 words (kept).

Run ``uv run python tests/fixtures/build_movies_sample.py`` to regenerate; a unit test asserts the committed CSV is
byte-identical to this script's output, so the fixture can never drift silently.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Any

import yaml

SEED = 20240607
N_ROWS = 300
N_SHORT = 8  # stub plots under 50 words: cleaning must drop them
EDGE_WORDS = {"edge49": 49, "edge50": 50}
HEADER = ["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"]
HERE = Path(__file__).parent
DATA = HERE / "story_data"
OUTPUT = HERE / "movies_sample.csv"

# origin -> (first names, last names, (earliest year, latest year), sampling weight). Names are invented-sounding.
ORIGINS: dict[str, tuple[list[str], list[str], tuple[int, int], int]] = {
    "American": (
        ["Walter", "Eleanor", "Marcus", "Dolores", "Vernon", "Cora", "Dale", "Wanda", "Lenore", "Hollis"],
        [
            "Hargrove",
            "Pruitt",
            "Delacroix",
            "Merrick",
            "Sutterfield",
            "Lindqvist",
            "Okonkwo",
            "Thibault",
            "Wexcombe",
            "Ostrander",
        ],
        (1915, 2017),
        40,
    ),
    "British": (
        ["Edmund", "Beatrice", "Alistair", "Margery", "Neville", "Philippa", "Rupert", "Winifred", "Gethin", "Imogen"],
        [
            "Ashworth",
            "Pemberton",
            "Thackeray",
            "Fairweather",
            "Lockhart",
            "Bramwell",
            "Cholmondeley",
            "Tregenza",
            "Quayle",
            "Haythorne",
        ],
        (1920, 2017),
        14,
    ),
    "Bollywood": (
        ["Aravind", "Meenal", "Rohit", "Sunita", "Vikrant", "Lata", "Kabir", "Anjali", "Devendra", "Nandini"],
        [
            "Joglekar",
            "Moitra",
            "Chitnis",
            "Dhawalkar",
            "Ranadive",
            "Bhosekar",
            "Kolhatkar",
            "Sengupta",
            "Pandharkar",
            "Vaidya",
        ],
        (1940, 2017),
        10,
    ),
    "Tamil": (
        ["Murugesan", "Kalyani", "Senthil", "Vasanthi", "Karthikeyan", "Lakshmi", "Pandian", "Revathi"],
        [
            "Annamalai",
            "Subramaniam",
            "Natarajan",
            "Chidambaram",
            "Velusamy",
            "Ayyadurai",
            "Sivakumaran",
            "Thirunavukkarasu",
        ],
        (1950, 2017),
        5,
    ),
    "Japanese": (
        ["Haruto", "Sakura", "Daisuke", "Kimiko", "Takeshi", "Yuki", "Shinji", "Michiko"],
        ["Tachibara", "Aokawa", "Ishizumi", "Hayakeyama", "Mizoguro", "Sakuradaira", "Tsugawara", "Hakamadajima"],
        (1925, 2017),
        7,
    ),
    "Hong Kong": (
        ["Wai", "Mei", "Chun", "Lan", "Kwok", "Siu", "Ping", "Yee"],
        ["Tsui", "Fong", "Chow", "Yip", "Ngai", "Kwan", "Lo", "Szeto"],
        (1960, 2017),
        4,
    ),
    "South_Korean": (
        ["Min-jun", "Seo-yeon", "Ji-ho", "Hye-jin", "Dong-hyun", "Eun-ji", "Tae-yang", "So-ra"],
        ["Bae", "Han", "Jo", "Oh", "Yoon", "Kwon", "Baek", "Seo"],
        (1960, 2017),
        4,
    ),
    "Australian": (
        ["Bruce", "Shelley", "Trevor", "Narelle", "Clive", "Jodie", "Gary", "Kerry"],
        ["Costain", "Yarrowood", "Dunmore", "Hargadon", "Mulgrave", "Tolhurst", "Ironbridge", "Pemberthy"],
        (1970, 2017),
        4,
    ),
    "Canadian": (
        ["Gilles", "Marguerite", "Hamish", "Solange", "Duncan", "Mireille", "Lorne", "Odette"],
        ["Gagnard", "Lavoisy", "Sutherby", "Wharnecliffe", "Beaudrin", "Chapdelaine", "Thibodault", "Charbonnet"],
        (1950, 2017),
        4,
    ),
    "Russian": (
        ["Dmitri", "Natalya", "Boris", "Irina", "Pavel", "Svetlana", "Anatoly", "Olga"],
        ["Beloselsky", "Verbitsky", "Dobrolyubov", "Rostovtsev", "Tomashevsky", "Zaitsev", "Kuznetsov", "Orlov"],
        (1925, 2017),
        4,
    ),
    "Turkish": (
        ["Emre", "Selin", "Kemal", "Aylin", "Burak", "Zeynep", "Cem", "Deniz"],
        ["Karabulut", "Yaşarel", "Gökbulut", "Tunçdemir", "Aksoylu", "Demirtaş", "Polatkan", "Özdilek"],
        (1955, 2017),
        2,
    ),
    "Malayalam": (
        ["Gopan", "Sreedevi", "Unni", "Anitha", "Radhakrishnan", "Maya", "Vijayan", "Thankamma"],
        ["Thulasidharan", "Varyar", "Ammal", "Vaidyar", "Kurup", "Thampi", "Panicker", "Warrier"],
        (1955, 2017),
        2,
    ),
}

# Earliest plausible release year for a genre label (so that, for example, no 1920 science-fiction film appears).
GENRE_MIN_YEAR = {"science fiction": 1950, "musical": 1930, "animation": 1930, "film noir": 1941}
MINOR_WORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "by", "with", "from"}


def _load(*names: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in names:
        merged.update(yaml.safe_load((DATA / name).read_text(encoding="utf-8")))
    return merged


def _load_anchors() -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for name in ("anchors_a.yaml", "anchors_b.yaml"):
        anchors += yaml.safe_load((DATA / name).read_text(encoding="utf-8"))
    return anchors


def _name(rng: random.Random, origin: str) -> str:
    firsts, lasts, _, _ = ORIGINS[origin]
    return f"{rng.choice(firsts)} {rng.choice(lasts)}"


def _cast(rng: random.Random, origin: str) -> str:
    names: list[str] = []
    target = rng.randint(2, 4)
    while len(names) < target:
        candidate = _name(rng, origin)
        if candidate not in names:
            names.append(candidate)
    return ", ".join(names)


def _wiki(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}_(synthetic_film)"


def _label_case(rng: random.Random, value: str) -> str:
    """Mimic the messy casing of the Kaggle file so that normalisation has something to do."""
    roll = rng.random()
    if roll < 0.10:
        return value.title()
    if roll < 0.12:
        return value.upper()
    return value


def title_from_object(obj: str) -> str:
    """``"the rope bridge"`` -> ``"The Rope Bridge"``: the central object names the film."""
    words = obj.replace("-", " - ").split()
    out = [w.capitalize() if i == 0 or w.lower() not in MINOR_WORDS else w.lower() for i, w in enumerate(words)]
    return " ".join(out).replace(" - ", "-")


def _fill(sentences: list[str], n: str, a: str, obj: str) -> str:
    """Join sentences and fill the placeholders; a character is named in full once, then by given name."""
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "obj":
            return obj
        full = n if key == "n" else a
        if key in seen:
            return full.split()[0]
        seen.add(key)
        return full

    return re.sub(r"\{(n|a|obj)\}", replace, " ".join(sentences))


def _plot(
    rng: random.Random, motif: dict[str, str], genre: str, origin: str, target: str, shared: dict[str, Any]
) -> str:
    """Compose a plot: the film's own premise and twist, beats about its own object, then a closing sentence."""
    n, a = _name(rng, origin), _name(rng, origin)
    while a.split()[0] == n.split()[0]:
        a = _name(rng, origin)
    if genre == "animation":  # cartoon characters get a given name only
        n, a = n.split()[0], a.split()[0]
    if target == "short":
        return _fill([motif["premise"]], n, a, motif["obj"])
    wanted = EDGE_WORDS.get(target)
    for _ in range(5000):
        beats = rng.sample(shared["beats"], rng.randint(0 if wanted else 2, 3))
        if wanted:  # short bridging sentences give the exact-length plots the granularity they need
            beats += rng.sample(shared["bridges"], rng.randint(0, 2))
        text = _fill(
            [motif["premise"], motif["turn"], *beats, rng.choice(shared["finales"][genre])], n, a, motif["obj"]
        )
        if wanted is None or len(text.split()) == wanted:
            return text
    raise ValueError(f"could not compose a {wanted}-word plot for {motif['obj']!r}")


def build_rows() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    shared = _load("shared.yaml")
    motifs_by_genre = _load("motifs_1.yaml", "motifs_2.yaml", "motifs_3.yaml")
    origins = list(ORIGINS)
    weights = [ORIGINS[o][3] for o in origins]

    rows: list[dict[str, str]] = []
    used_titles: set[str] = set()
    for anchor in _load_anchors():
        used_titles.add(anchor["title"])
        rows.append(
            {
                "Release Year": str(anchor["year"]),
                "Title": anchor["title"],
                "Origin/Ethnicity": anchor["origin"],
                "Director": anchor["director"],
                "Cast": anchor["cast"],
                "Genre": anchor["genre"],
                "Wiki Page": _wiki(anchor["title"]),
                "Plot": " ".join(anchor["plot"].split()),
            }
        )

    pool = [(genre, motif) for genre, motifs in motifs_by_genre.items() for motif in motifs]
    rng.shuffle(pool)
    n_generated = N_ROWS - len(rows)
    chosen = pool[:n_generated]  # drawn without replacement: no two films share a motif
    # The exact-length edge plots go to the motifs with the shortest premise and twist (49 and 50 words are reachable).
    by_length = sorted(range(n_generated), key=lambda i: len((chosen[i][1]["premise"] + chosen[i][1]["turn"]).split()))
    rest = ["short"] * N_SHORT + ["normal"] * (n_generated - N_SHORT - len(EDGE_WORDS))
    rng.shuffle(rest)
    edge_at = dict(zip(by_length, EDGE_WORDS, strict=False))
    kinds = [edge_at[i] if i in edge_at else rest.pop() for i in range(n_generated)]
    for (genre, motif), kind in zip(chosen, kinds, strict=True):
        origin = rng.choices(origins, weights)[0]
        lo, hi = ORIGINS[origin][2]
        title = title_from_object(motif["obj"])
        if title in used_titles:
            raise ValueError(f"duplicate title {title!r}: give the motif a more specific object")
        used_titles.add(title)
        genre_cell = _label_case(rng, genre)
        origin_cell = origin
        if rng.random() < 0.04:
            genre_cell = "unknown"
        if rng.random() < 0.04:
            origin_cell = "Unknown"
        rows.append(
            {
                "Release Year": str(rng.randint(max(lo, GENRE_MIN_YEAR.get(genre, lo)), hi)),
                "Title": title,
                "Origin/Ethnicity": origin_cell,
                "Director": _name(rng, origin) if rng.random() > 0.05 else "Unknown",
                "Cast": _cast(rng, origin),
                "Genre": genre_cell,
                "Wiki Page": _wiki(title),
                "Plot": _plot(rng, motif, genre, origin, kind, shared),
            }
        )
    rng.shuffle(rows)
    return rows


def write_csv(path: Path = OUTPUT) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows())


if __name__ == "__main__":
    write_csv()
    print(f"wrote {OUTPUT}")
