# Backlog

Follow-ups moved out of PRs. One bullet per item: `- [PR-NN] description`.
- [PR-01] `make eval-smoke` / `ci` retrieval half are stubs until PR-09; install gitleaks locally to exercise the scan in `make ci`.
- [PR-02] `docs/CREDENTIALS.md` is referenced by every credential hint but is written in PR-11; until then the hints point at a missing file.
- [PR-02] No public no-auth mirror of the Kaggle dataset was adopted (licence unclear: CC BY-SA vs CC BY-NC-SA reports); revisit if a mirror with a verified licence and identical columns appears.
- [PR-02] `.env.example` (unreadable by the builder role) should list the optional overrides `QDRANT_URL`, `NEBIUS_BASE_URL`, `MCP_URL`, `LANGSMITH_PROJECT` if it does not already.
- [PR-02] Kaggle auth supports only `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars, not `~/.kaggle/kaggle.json` or the newer single-token form.
- [PR-02] Fixture plots are distinctive but formulaic (one premise sentence, one twist, object-linked beats); PR-08 may want to hand-write more of them if fuzzy questions still prove ambiguous.
- [PR-02 QA n1] Credential-key filter in observability no longer rejects names like `github_token`, `hf_token`, `openai_key`, `passphrase`; value check still catches configured secrets. Broaden the name heuristic.
- [PR-02 QA n2] Fixture plots repeat each film's object (also its title) 2–4×, giving BM25 an easy match. PR-08 fuzzy questions must paraphrase the object, not name it.
- [PR-02 QA n3] Anchor director "Boris Zaitsev" matches a well-known real writer; rename to an invented name.
- [PR-03] A point whose chunk text changed keeps its id, so resume-by-id does not refresh it; changing chunking parameters or data needs `make ingest RECREATE=1`. A content-hash payload field would allow selective refresh.
- [PR-03] Sentence splitting is a regex: it only splits before an uppercase letter, digit or quote and uses a short abbreviation list. Lowercase-initial sentences (rare in Wikipedia plots) are not split; swap in a real splitter if the full dataset shows bad chunks.
- [PR-03] Ingest embeds on CPU in 256-chunk batches (about 27 s for the 304-chunk fixture including model load); the full 35k-film dataset is untimed. Consider FastEmbed `parallel`/threads if a full ingest is too slow.
- [PR-03] The integration tests fall back to `QdrantClient(":memory:")` when `QDRANT_URL` is unset (so `make check` works offline); payload-index assertions only run against the real service (CI).
- [PR-03 QA m1] Sentence splitter misses boundaries like "say no. The", "Plan B. The", "U.S. There" (`ingest/chunk.py:36-40`).
- [PR-03 QA m2] Chunk overlap can be 0 when the last sentence alone exceeds 40 tokens (`ingest/chunk.py:127-131`).
- [PR-03 QA m3] The `:memory:` fallback in `tests/integration/test_ingest_qdrant.py` (lines 35, 91) gives no signal that payload-index checks were skipped.
- [PR-03 QA m4] Token counting uses an internal FastEmbed call (`ingest/embed.py:83`) with no upper version pin on fastembed.
- [Orchestrator] `.claude/settings.json` deny rule `Read(./.env.*)` also blocks reading `.env.example`; human decision pending on adding an allow rule.
- [Orchestrator] CI pulls `qdrant/qdrant` anonymously and hits Docker Hub rate limits (PR #3 needed many reruns); human to add `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets and `credentials:` on the qdrant services.
