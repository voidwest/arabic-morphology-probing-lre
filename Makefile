PYTHON ?= python3
BF16_MATRIX ?= configs/bf16_model_matrix_20260826_jais_resolved.json
BF16_FREEZE ?= evidence/bf16-analysis-20260826-v2
BF16_RUN_ROOT ?= runs/bf16-analysis
BF16_RUN_ID ?=
BF16_SYNC_DEST ?=

.PHONY: test data-verify freeze-verify paper data-rebuild bf16-preflight \
        bf16-queue-init bf16-queue-run bf16-queue-status

test:
	$(PYTHON) -m pytest -q tests

data-verify:
	$(PYTHON) scripts/data/verify_paper1_dataset.py

freeze-verify:
	$(PYTHON) scripts/verify_bf16_analysis_freeze.py $(BF16_FREEZE) --skip-external

paper:
	cd paper/paper1_lre_revision && \
	pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
	bibtex main && \
	pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

data-rebuild:
	$(PYTHON) scripts/data/rebuild_paper1_dataset.py

bf16-preflight:
	$(PYTHON) scripts/bf16_preflight.py --manifest $(BF16_MATRIX) --run-root $(BF16_RUN_ROOT)

bf16-queue-init:
	@test -n "$(BF16_RUN_ID)" || (echo 'set BF16_RUN_ID to a new unique ID' >&2; exit 2)
	$(PYTHON) scripts/bf16_queue.py queue-init --manifest $(BF16_MATRIX) \
		--run-root $(BF16_RUN_ROOT)/$(BF16_RUN_ID) --run-id $(BF16_RUN_ID)

bf16-queue-run:
	@test -n "$(BF16_SYNC_DEST)" || (echo 'set BF16_SYNC_DEST explicitly' >&2; exit 2)
	@test -n "$(BF16_RUN_ID)" || (echo 'set BF16_RUN_ID to the existing run ID' >&2; exit 2)
	$(PYTHON) scripts/bf16_queue.py queue-run --manifest $(BF16_MATRIX) \
		--run-root $(BF16_RUN_ROOT)/$(BF16_RUN_ID) --sync-destination $(BF16_SYNC_DEST)

bf16-queue-status:
	@test -n "$(BF16_RUN_ID)" || (echo 'set BF16_RUN_ID to the existing run ID' >&2; exit 2)
	$(PYTHON) scripts/bf16_queue.py queue-status --run-root $(BF16_RUN_ROOT)/$(BF16_RUN_ID)
