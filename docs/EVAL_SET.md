# Evaluation set (v1)

Forty questions, ten per type, used by the evaluation runner (PR-09) to compare dense, sparse and hybrid retrieval and
to score the agent. File: `src/movie_rag/eval/data/questions_v1.jsonl`. Loader and checks:
`src/movie_rag/eval/questions.py` (`load_eval_set()`).

## Schema

One JSON object per line, validated by `EvalQuestion`:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | unique, `<type-prefix>-NN` (`fuzzy-01` ... `unanswerable-10`) |
| `type` | `fuzzy_plot` / `exact_entity` / `filtered` / `unanswerable` | see below |
| `question` | string | what the user asks, in the user's words |
| `gold_movie_ids` | list of strings | `movie_id` of the film that answers it (empty for `unanswerable`) |
| `filters` | object | `year_from`, `year_to`, `genre`, `origin` as accepted by `search_movies` (empty `{}` when none) |
| `absent_title` | string or absent | `unanswerable` only: the plausible film that is not in the index |
| `notes` | string | why the question exists and what it tests |

`movie_id` is `slug(title)-year-row_index` (see `ingest/clean.py`), so a gold id is only valid for the file it was
written from: this set is written against `tests/fixtures/movies_sample.csv`.

## Types

| Type | What it asks | Gold | Right answer |
|---|---|---|---|
| fuzzy plot | a film described by its premise in other words | one film | the film appears in the retrieved list and is cited |
| exact entity | a film by its title or its director | one film | the named film |
| filtered | a premise plus year range, genre and/or origin | one film that satisfies the filters | the film, found inside the filter |
| unanswerable | a plausible film that is not in the index | none | abstain (no citation) |

Cast members are deliberately not used for exact-entity questions: ingestion prefixes each chunk with
`Title (Year) | Genre | Director`, so the cast is stored as metadata but is not searchable text.

## How the set was written

The plan is to generate the set by paraphrasing sampled plots with the chat model. No `NEBIUS_API_KEY` exists yet,
so the 40 questions were written by hand from the fixture films, to the same rules the generator enforces:

* **Fuzzy questions paraphrase the film's distinctive object** instead of naming it (a "church bell" becomes "a chime
  from the steeple", "a crown of reeds" becomes "a headpiece woven from rushes"). The fixture plots repeat that object
  two to four times and it is usually the title, so naming it would let BM25 win trivially (backlog PR-02 QA n2).
  Five have long plots that the chunker splits (`fuzzy-01`, `-03`, `-04`, `-09`, `-10`), five have short plots.
* **The overlap guard**: no run of four words (lowercased, punctuation stripped, `eval.ngram_size`) shared with the
  film's plot. It is a function (`eval/overlap.py`), run over the fuzzy questions by
  `test_every_fuzzy_question_passes_the_overlap_guard`. The same guard and a no-title check are also applied to the
  filtered questions, and to the unanswerable questions against every plot in the fixture.
* **Filtered questions** describe the premise in other words and carry the filter that the gold film satisfies; the
  check `matches_filters` uses the semantics of the retrieval layer (inclusive year range, exact lowercase genre and
  origin). Three filters isolate the gold film on their own (`filtered-01`, `-03`, `-04`); the other seven let two to six films
  through, so the premise has to do the work.
* **Exact questions**: six name a title, four a director. The check requires the question to contain the title or
  the director of its gold film. Directors named here are unique in the fixture. The anchor director "Boris Zaitsev"
  (a name close to a real writer, backlog PR-02 QA n3) is never used in a question; he directs two films.
* **Unanswerable questions** describe films that are not in the fixture: three are easy (nothing close), five are near
  misses that share a setting or object with fixture films (trench, heist, radio, bees, lighthouse), and two carry
  filters that match no film (one also echoes a dentist from a fixture western). `absent_title` names the imagined film and must not be a fixture title.
* Each gold film is the answer to at most one question.

Questions that the guard rejected while writing (rewritten before commit): `fuzzy-04` ("to a hidden city"),
`filtered-04` ("six strangers recruited by"), `filtered-10` ("in a mountain village"), `unanswerable-09` ("falls in love
with", which appears in two fixture plots). QA review of all 40 is recorded in `reports/qa/`.

## Generator (needs a key)

`python -m movie_rag.eval.generate` samples films with `ingest.random_seed`, asks the chat model for a paraphrase with
the versioned prompt `src/movie_rag/eval/prompts/paraphrase_v1.md`, rejects an answer that shares a four-word run with
the plot or contains the title, and asks again (up to `eval.max_attempts`) with the offending phrases listed. A film
that never yields a clean question is reported as rejected and the next sampled film is tried. It writes fuzzy
questions to `eval.generated_path` (never over `questions_v1.jsonl`); a person reviews them before they replace
anything. Without `NEBIUS_API_KEY` it prints `set NEBIUS_API_KEY — see docs/CREDENTIALS.md` and exits 2. It is tested
with a scripted chat model.

## LangSmith dataset

`python -m movie_rag.eval.upload` creates the dataset `eval.dataset_name` (`movie-plots-questions-v1`) with one example
per question (inputs: `question`, `filters`; outputs: `gold_movie_ids`, `expect_abstention`; metadata: `question_id`,
`type`, `absent_title`). Without `LANGSMITH_API_KEY` it prints `skipped: no LANGSMITH_API_KEY ...` and exits 0. An
existing dataset of that name is left untouched; change the name in `config.yaml` when the questions change.

## Questions

| id | type | question | gold film (`movie_id`) | filters |
|---|---|---|---|---|
| `fuzzy-01` | fuzzy_plot | A doctor takes over the practice of a vanished colleague in an isolated Yorkshire hamlet, where a chime from the steeple always foretells a death, and finds he had sketched one more chime that has never sounded. | The Ninth Bell of Varnholt (1931), `the-ninth-bell-of-varnholt-1931-14` | - |
| `fuzzy-02` | fuzzy_plot | A private eye comes round on a rainy waterfront with a grazed skull and no idea of his own name, and starts to fear that the man he has been hired to track down is himself. | The Forgetting Hour (1947), `the-forgetting-hour-1947-282` | - |
| `fuzzy-03` | fuzzy_plot | In a drought-stricken outback town, a teenage mechanic keeps tall transparent farming towers running that pull moisture out of the night air, until she learns the water is secretly supplied by the mining firm whose blasting is damaging the ground. | Glass Harvest (2013), `glass-harvest-2013-198` | - |
| `fuzzy-04` | fuzzy_plot | A lonely accountant takes a late train on a stormy night and reaches a concealed city where every anniversary of birth that nobody marked ends up archived, and has to persuade strangers to throw her parties to get her years back. | Atlas of Lost Birthdays (2015), `atlas-of-lost-birthdays-2015-248` | - |
| `fuzzy-05` | fuzzy_plot | An engineer on a vessel carrying settlers across a desert planet finds that the reprocessed drinking supply carries the faint patterns of a tongue nobody aboard speaks. | The Recycled Water Signal (1993), `the-recycled-water-signal-1993-8` | - |
| `fuzzy-06` | fuzzy_plot | A skyscraper window washer glimpses a killing through the pane on a high floor, and the murderer notices her back; a lawyer from the storey beneath tries to buy her silence. | The Window-Cleaner's Witness (1997), `the-window-cleaner-s-witness-1997-6` | - |
| `fuzzy-07` | fuzzy_plot | A keeper of a mountain refuge opens the visitors' register to find tomorrow already filled in by a walker who has not shown up yet. | The Pre-Signed Guest Book (1957), `the-pre-signed-guest-book-1957-155` | - |
| `fuzzy-08` | fuzzy_plot | A trainee caring for a baby fire-breather learns that it lives solely on melodies, and the village's choir director finds it has swallowed their finest hymn. | The Song-Eating Hatchling (1992), `the-song-eating-hatchling-1992-103` | - |
| `fuzzy-09` | fuzzy_plot | A farmer's daughter in the fenland jokingly puts on a headpiece woven from rushes she found drifting in a flood, and the wetlands start obeying her while she gradually forgets her own name. | The Hollow Crown of Marsh End (1996), `the-hollow-crown-of-marsh-end-1996-113` | - |
| `fuzzy-10` | fuzzy_plot | An insurer hires a deep-sea diver to locate a freighter lost near a famous Australian coral reef, but inside he finds the hull was cut open from within and a hidden logbook showing the vessel was sunk deliberately. | Salvage Rights (1981), `salvage-rights-1981-22` | - |
| `exact-01` | exact_entity | Tell me about the film Counterfeit Spring. | Counterfeit Spring (1952), `counterfeit-spring-1952-151` | - |
| `exact-02` | exact_entity | What is the plot of Kimchi Wars? | Kimchi Wars (2014), `kimchi-wars-2014-114` | - |
| `exact-03` | exact_entity | Which film is Nile Postman and when was it released? | Nile Postman (1960), `nile-postman-1960-43` | - |
| `exact-04` | exact_entity | Find the film titled Pashmina. | Pashmina (1998), `pashmina-1998-93` | - |
| `exact-05` | exact_entity | What happens in Echo Chamber 9? | Echo Chamber 9 (2016), `echo-chamber-9-2016-136` | - |
| `exact-06` | exact_entity | Look up Black Tulip Hotel for me. | Black Tulip Hotel (1971), `black-tulip-hotel-1971-190` | - |
| `exact-07` | exact_entity | Which film did Gleb Ostrovsky-Marin direct? | Rails Beneath the Snow (1963), `rails-beneath-the-snow-1963-277` | - |
| `exact-08` | exact_entity | What did the director Ling Wai-hung make? | Midnight Ferry to Kowloon (1988), `midnight-ferry-to-kowloon-1988-250` | - |
| `exact-09` | exact_entity | Show me the film directed by Rejean Thibodault. | Tin Soldiers of Avenue Dorval (1969), `tin-soldiers-of-avenue-dorval-1969-227` | - |
| `exact-10` | exact_entity | Is there a film by the director Maxime Delorimier-Haas, and what is it about? | Cold Storage (2017), `cold-storage-2017-260` | - |
| `filtered-01` | filtered | A film about villagers in a dry region who hijack a fuel truck to expose a rich landowner who is draining their well. | Kerosene Summer (1975), `kerosene-summer-1975-49` | origin=telugu |
| `filtered-02` | filtered | A British wartime comedy in which officials try to convince the enemy that the invasion will come ashore at a seaside resort by staging a mock social gathering on the sand. | Operation Teacup (1944), `operation-teacup-1944-297` | year_from=1940, year_to=1949, origin=british |
| `filtered-03` | filtered | A widow leads thousands of cattle hundreds of kilometres through the outback to a railway, betrayed by the man who hired her. | The Drover's Hymn (1979), `the-drover-s-hymn-1979-213` | genre=western, origin=australian |
| `filtered-04` | filtered | Half a dozen strangers hired by an ex-illusionist plan a vault robbery while flooding paralyses the city. | Monsoon Heist (2005), `monsoon-heist-2005-280` | year_from=2000, year_to=2009, genre=crime, origin=bollywood |
| `filtered-05` | filtered | Textile workers organise a choir whose songs plead with the owner's family to keep the factory open. | A Song for the Sleeping Mill (1977), `a-song-for-the-sleeping-mill-1977-185` | genre=musical, origin=tamil |
| `filtered-06` | filtered | Travellers marooned at a provincial railway halt confess their secrets to one another while the keeper's spouse starts to doubt she wants a train to turn up. | Seven Days of Rain in Tsarskoye (1927), `seven-days-of-rain-in-tsarskoye-1927-253` | year_from=1920, year_to=1929, origin=russian |
| `filtered-07` | filtered | A tiny seafaring vessel made from a nut husk crosses a kitchen basin, chased by a buccaneer made of sponge. | The Walnut-Shell Boat (1968), `the-walnut-shell-boat-1968-10` | year_from=1960, year_to=1969, genre=animation |
| `filtered-08` | filtered | Every rider on the late bus holds the same dark parasol, and the driver grows uneasy. | The Identical Umbrellas (1997), `the-identical-umbrellas-1997-269` | genre=thriller, origin=hong kong |
| `filtered-09` | filtered | On a failing orbital outpost a botanist protects her apple grove from being jettisoned, because it secretly keeps the air breathable. | Orbit of the Last Orchard (1984), `orbit-of-the-last-orchard-1984-91` | genre=science fiction, origin=russian |
| `filtered-10` | filtered | An elderly ceramicist in a village high in the hills promises her workshop to whichever apprentice can produce seven identical tea cups before winter's first snowfall. | Seven Stones for Hatsue (1956), `seven-stones-for-hatsue-1956-40` | year_from=1950, year_to=1959, origin=japanese |
| `unanswerable-01` | unanswerable | Find the film where a talking dinosaur runs for president of Mars. | none (absent: *Raptor for President*) | - |
| `unanswerable-02` | unanswerable | I remember a film in which submariners discover a civilisation of intelligent jellyfish living in an undersea trench. | none (absent: *The Jellyfish Court*) | - |
| `unanswerable-03` | unanswerable | Which film is about a retired figure skater who assembles a crew to rob a diamond exchange in 1970s Rotterdam? | none (absent: *Triple Axel Heist*) | - |
| `unanswerable-04` | unanswerable | What was that movie where a time-travelling samurai opens a sushi franchise in Texas? | none (absent: *Ronin Roll*) | - |
| `unanswerable-05` | unanswerable | A teenage gang starts a pirate radio station inside an abandoned zoo in 1980s Brazil, what is it called? | none (absent: *Frequency Zoo*) | - |
| `unanswerable-06` | unanswerable | Is there a mockumentary about a Scandinavian curling team that turns into a cult sensation? | none (absent: *Sweep Nation*) | - |
| `unanswerable-07` | unanswerable | An astronaut stranded on Venus teaches a swarm of bees to play baseball, which film is that? | none (absent: *Hive Run*) | - |
| `unanswerable-08` | unanswerable | Recommend a silent whaling drama from Norway. | none (absent: *The Harpoon Winter*) | year_from=1920, year_to=1929, origin=norwegian |
| `unanswerable-09` | unanswerable | I am looking for the film about a sentient lighthouse that becomes infatuated with a passing cargo ship. | none (absent: *Beam of Devotion*) | - |
| `unanswerable-10` | unanswerable | A vampire dentist in 1950s Mumbai, what is the film? | none (absent: *Fangs and Fillings*) | year_from=1950, year_to=1959, genre=horror, origin=bollywood |

## Known limitations

* **Fixture only.** The gold ids belong to the 300-film synthetic fixture. The real 35k-film dataset needs its own
  set (the generator is for that), and results on this one say nothing about it.
* **Easy for lexical search.** The fixture plots are short, formulaic and unique per film, so a paraphrase still
  shares rare single words with its plot. An informal check in a scratch script (not an evaluation result, not in any
  report) put the gold film at rank 1 in BM25 mode for all ten fuzzy questions, and in the top 8 for 9 of 10 in dense
  mode; the PR-09 report is the authoritative source. Do not expect the fixture to separate the three modes much.
* **Hand-written, so subjective.** Another author would phrase the paraphrases differently; the guard only bounds
  copying, not difficulty.
* **One gold film per question.** Filtered questions can have other films that also fit; they count as misses.
* **No cast questions** (cast is not indexed text), no multi-film questions ("two films by the same director"), no
  follow-up turns.
* **Abstention near misses are few.** Five of the ten unanswerable questions are near misses; a larger set would
  separate a model that abstains from one that merely retrieves nothing.
