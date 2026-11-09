# ARC synthetic tasks: 1,000 per-task packages

This release contains all 1,000 targeted per-task authored candidate packages; the file count is complete. Below are, separately, the full-batch replay, the exact-version limited independent review, and retained historical evidence; reaching the count is not in itself proof of batch-wide semantic uniqueness or of original-ARC difficulty.

Each package contains its own transform.py, generate_input.py, task.json, Traditional Chinese rules, parent sources, and the author's verification record. At least two parents, at least one from evaluation; sources pinned at arc-forge 55a85ef8a3e782419ac6d2c0c1c1b24474b194ae.

In this release 102 packages exactly match the version of their limited independent review files, and 1 has a version difference. The records include 34 reviews newly done after recovery and 69 retained earlier evidence; retained evidence does not claim to have been retested this time. See REVIEW_STATUS.json. A limited sample does not prove global correctness, a unique rule, novelty, or original-ARC difficulty.

Replaced task IDs contain only the current version; replacements and test-case fixes keep only the current version. Scope limits and handled issues are in OPEN_REVIEW_NOTES.json and RESOLVED_REVIEW_NOTES.json.

After unpacking, open index.html to browse all rules, parents, review status, and inputs/outputs offline; source links all point inside the package.

Uses Python and NumPy; after installing requirements.txt run:

    python validate_examples.py --jobs 4
    python validate_examples.py --jobs 4 --generated 5

The first replays the stored answers; the second additionally runs five generator replays and format checks per task, adding no independent expected-answer evidence. The program does not modify tasks. The full-batch measurement is in REPLAY_SUMMARY.json; per-file checksums in FILE_SHA256.json.

Snapshot: 20261005T192835Z

Tested runtime: Python 3.12.14, NumPy 2.3.5. Some early compact syntax emits SyntaxWarning in this environment; the full-batch verification still completed. These results are not extrapolated into a compatibility guarantee for all future Python versions. The SciPy version for the optional independent checks is described separately.

## Optional independent checks

This release also includes exact-version, limited-sample pixel check programs for 34 tasks; this is not full-batch acceptance. Dependencies, version locks, isolated output, and how to run are in independent_checks/README.md. Historical reviews and unlisted tasks are not replayed by this set.

The offline browser code has been run in a minimal DOM model over all 4,000 task / split / answer display combinations, checking local code links, search, and status filters. This is a JavaScript functional check and does not claim that visual layout testing in a real browser was done; details in VIEWER_VALIDATION.json.

The scope and counts of this release's mechanical comparison are in MECHANICAL_AUDIT_SCOPE.json; the coverage of stored grids and of program ASTs is listed separately.

Full-batch measurement of this release: 8813 stored-answer replays and 5000 generator replays / format checks, all passed; the latter is not new independent expected-answer verification.
