# ADR 002: Server-side RRF, fixed constant, prefetch and grouping parameters

* Status: Accepted; the fixed RRF constant is to be revisited with qdrant-client 1.16 (backlog)
* Date: 2026-10-10

## Context

Dense vectors find paraphrases; BM25 finds rare words. Their scores are not comparable (cosine similarity versus an
unbounded BM25 sum), so combining them needs either rank-based fusion or score normalisation. The product spec asks for
two prefetches fused with RRF in one Query API call, filters inside each prefetch, one row per film, and every parameter
in `config.yaml`.

## Decision

`retrieval/search.py: build_request` builds, for `hybrid`:

```text
query_points_groups(
  prefetch=[Prefetch(dense,  using="dense", limit=50, filter=F),
            Prefetch(bm25,   using="bm25",  limit=50, filter=F)],
  query=FusionQuery(fusion=Fusion.RRF),
  group_by="movie_id", group_size=1, limit=top_k)
```

| Parameter | Value (`config.yaml`) | Notes |
|---|---|---|
| `retrieval.prefetch_limit` | 50 | chunks per branch, not films; spec value, not tuned |
| `retrieval.top_k` | 8 | default films returned; `mcp.max_top_k: 20` bounds the tool argument, below the prefetch limit |
| `retrieval.default_mode` | `hybrid` | `dense` and `sparse` use the same function with one named vector |
| RRF constant `k` | **not configurable; effective value 2** | see below |
| grouping | `group_by=movie_id`, `group_size=1` | each film once, represented by its best fused chunk |

**The RRF constant.** Qdrant fuses with `score = sum over lists of 1 / (k + rank)` with a 0-based rank. With plain
`Fusion.RRF` the server uses `k = 2` (the documented default), not the usual 60: the PR-04 QA check against Qdrant
v1.15.4 saw a top-ranked chunk score 0.5 in plain mode and 1/60 when `{"rrf": {"k": 60}}` was sent over REST. A
configurable `k` is a Qdrant feature "as of v1.16.0" (weighted RRF as of v1.17.0), and qdrant-client 1.15.1 has no
`RrfQuery` model to send it (the PR-04 QA probe found that server 1.15.4 already accepts the field over REST, but that is outside the supported client API and we do not rely on it). We therefore did not invent a `retrieval.rrf_k` setting that would be silently ignored.
Consequence of `k = 2`: ranks are weighted steeply (rank 0 contributes 0.5, rank 1 0.333, rank 2 0.25), so a film
that is first in one list and absent from the other scores below a film that is second in both lists.

**Filters** (`year_from`, `year_to`, `genre`, `origin`) are attached to each prefetch. A filter applied only to the
fused query would take its candidates from an unfiltered top 50 and could leave fewer than `top_k` rows.

**Ties.** Equal fused scores are common (a film first in one list and second in the other ties with its mirror image),
and Qdrant breaks ties *inside* each prefetch list (BM25 scores tie on near-identical plots) before fusing. The client
orders exactly equal output scores by `movie_id` (`by_score_then_id`), but that does **not** make hybrid results
repeatable: 15 identical calls produced two different result sets in PR-09. The evaluation metrics are tie-aware to
absorb this (`docs/EVALUATION.md`); a stable ranking would need client-side fusion with a stable tie-break.

## Alternatives considered

| Option | Why not now |
|---|---|
| Parametric RRF (`k = 60` or tuned) | Right direction; blocked by qdrant-client 1.15.1 (server 1.15.4 accepts it over REST, PR-04 QA). Needs a client bump to 1.16 or later and a `retrieval.rrf_k` setting. |
| DBSF (`Fusion.DBSF`, distribution-based score fusion) | Uses scores, so it depends on score distributions that differ between cosine and BM25; not evaluated. |
| Weighted score fusion with tuned weights | Needs a labelled set from the real dataset; the only labelled set is synthetic and favours BM25 (see `docs/EVALUATION.md`), so weights tuned on it would be wrong. |
| Client-side fusion | Gives a stable tie-break and any formula, at the cost of two round trips and our own filter/grouping code. Candidate if repeatability matters. |
| Dense-only or sparse-only | Kept as modes (they are the baselines of the evaluation), not as the default. |

## Consequences

* Whether hybrid beats its parts is **not demonstrated by the numbers in this repository**: the synthetic fixture favours
  BM25 and has 30 answerable questions (README "Results", `reports/EVAL_RESULTS.md`). The defaults above are the spec's, not
  tuned values.
* `prefetch_limit` counts chunks: a long film with several matching chunks uses several slots of a prefetch list.
* Sparse mode returns fewer than `top_k` films when few chunks share a term with the query (BM25 gives non-matching
  chunks no score); hybrid then fills from the dense list.
* Changing `k` later changes every hybrid score and rank; re-run `make eval` and `make report` and commit the refreshed
  reports.

## References

* RRF, constant `k` and weights: https://qdrant.tech/documentation/concepts/hybrid-queries/
* `docs/BACKLOG.md` entries `[PR-04] rrf_k`, `[PR-04 QA m1]`, `[PR-09]` (ties); `reports/qa/pr-4.md` finding m1.
