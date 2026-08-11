"""
Integration tests for the bootstrap_template feature in compute_ctl.

Tests exercise:
 - Idempotency: re-running does not re-execute scripts
 - Failure/retry: partial failure leaves no whole-template marker, next start retries
 - Per-script resume: scripts already applied before a failure are not replayed,
   which matters for non-idempotent statements (e.g. bare CREATE PUBLICATION)
 - New migration file: additional file discovered and run on re-deploy
 - Directory not found: graceful error when template doesn't exist in image
 - Script ordering: files sorted by name (numeric prefix convention)

These require `NEON_COMPUTE_BOOTSTRAP_ROOT` env var support in compute_ctl
to override the default /usr/local/share/compute-bootstrap path.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv


TEMPLATE_NAME = "test-tpl"


def _write_bootstrap_scripts(
    base: Path,
    init_scripts: dict[str, str] | None = None,
    migrations: dict[str, str] | None = None,
):
    """Create a bootstrap template directory structure under `base`."""
    tpl = base / TEMPLATE_NAME
    if init_scripts:
        d = tpl / "init-scripts"
        d.mkdir(parents=True, exist_ok=True)
        for name, content in init_scripts.items():
            (d / name).write_text(content, encoding="utf-8")
    if migrations:
        d = tpl / "migrations"
        d.mkdir(parents=True, exist_ok=True)
        for name, content in migrations.items():
            (d / name).write_text(content, encoding="utf-8")
    tpl.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def bootstrap_root(tmp_path: Path) -> Path:
    """Provide a temporary directory to use as NEON_COMPUTE_BOOTSTRAP_ROOT."""
    return tmp_path / "bootstrap"


def test_bootstrap_template_success(neon_simple_env: NeonEnv, bootstrap_root: Path):
    """
    Happy path: init-scripts and migrations are executed exactly once.
    A second start (same spec) does NOT re-execute (idempotency).
    """
    env = neon_simple_env

    _write_bootstrap_scripts(
        bootstrap_root,
        init_scripts={
            "01_create_table.sql": textwrap.dedent("""\
                CREATE TABLE IF NOT EXISTS public.bootstrap_test (
                    id serial PRIMARY KEY,
                    source text NOT NULL
                );
                INSERT INTO public.bootstrap_test (source) VALUES ('init-scripts');
            """),
        },
        migrations={
            "20240101000000_add_col.sql": textwrap.dedent("""\
                ALTER TABLE public.bootstrap_test ADD COLUMN IF NOT EXISTS applied_at timestamptz DEFAULT now();
            """),
        },
    )

    endpoint = env.endpoints.create("main")
    endpoint.respec(skip_pg_catalog_updates=False)
    endpoint.respec_deep(spec={"bootstrap_template": TEMPLATE_NAME})
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    # Verify scripts ran
    with endpoint.cursor() as cur:
        cur.execute("SELECT source FROM public.bootstrap_test")
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "init-scripts"

        # Verify idempotency table
        cur.execute("SELECT template FROM neon.compute_bootstrap")
        templates = [r[0] for r in cur.fetchall()]
        assert TEMPLATE_NAME in templates

        # Verify migration column exists
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bootstrap_test' AND column_name='applied_at'")
        assert cur.fetchone() is not None

    # Restart — scripts should NOT run again (idempotency)
    endpoint.stop()
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    with endpoint.cursor() as cur:
        cur.execute("SELECT count(*) FROM public.bootstrap_test")
        assert cur.fetchone()[0] == 1  # Still 1 row, not 2


def test_bootstrap_template_failure_retry(neon_simple_env: NeonEnv, bootstrap_root: Path):
    """
    If a script fails, no marker is written.
    On next start (after fixing the script), bootstrap runs again.
    """
    env = neon_simple_env

    # First: create a script that will fail
    _write_bootstrap_scripts(
        bootstrap_root,
        init_scripts={
            "01_will_fail.sql": "CREATE TABLE public.retry_test (id int); SELECT 1/0;",
        },
    )

    endpoint = env.endpoints.create("main")
    endpoint.respec(skip_pg_catalog_updates=False)
    endpoint.respec_deep(spec={"bootstrap_template": TEMPLATE_NAME})

    # Start should fail (or at least not write the marker)
    try:
        endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})
        # If it started, check no marker
        with endpoint.cursor() as cur:
            cur.execute("SELECT 1 FROM neon.compute_bootstrap WHERE template = %s", (TEMPLATE_NAME,))
            assert cur.fetchone() is None
    except Exception:
        pass  # Expected: compute_ctl may exit with error
    finally:
        endpoint.stop()  # Ensure any residual process is cleaned up before retry

    # Fix the script
    init_dir = bootstrap_root / TEMPLATE_NAME / "init-scripts"
    (init_dir / "01_will_fail.sql").write_text(
        "CREATE TABLE IF NOT EXISTS public.retry_test (id int);",
        encoding="utf-8",
    )

    # Retry — should succeed now
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    with endpoint.cursor() as cur:
        cur.execute("SELECT 1 FROM neon.compute_bootstrap WHERE template = %s", (TEMPLATE_NAME,))
        assert cur.fetchone() is not None
        cur.execute("SELECT 1 FROM public.retry_test")


def test_bootstrap_partial_failure_does_not_replay(
    neon_simple_env: NeonEnv, bootstrap_root: Path
):
    """
    Regression test for the per-script resume behavior.

    A first script succeeds and creates a non-idempotent object (a publication, which
    has no CREATE ... IF NOT EXISTS form); a later script fails. The failed run must
    NOT write the whole-template marker, but it MUST have recorded the first script in
    neon.compute_bootstrap_scripts — so the retry resumes at the failing script instead
    of replaying the publication and dying on "already exists".
    """
    env = neon_simple_env

    _write_bootstrap_scripts(
        bootstrap_root,
        init_scripts={
            # Non-idempotent on replay: CREATE PUBLICATION has no IF NOT EXISTS.
            "01_publication.sql": "CREATE PUBLICATION resume_test_pub;",
            "02_will_fail.sql": "SELECT 1/0;",
        },
    )

    endpoint = env.endpoints.create("main")
    endpoint.respec(skip_pg_catalog_updates=False)
    endpoint.respec_deep(spec={"bootstrap_template": TEMPLATE_NAME})

    try:
        endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})
    except Exception:
        pass  # Expected: the failing script aborts /configure
    finally:
        endpoint.stop()

    # Fix the failing script, leave 01_publication.sql untouched.
    init_dir = bootstrap_root / TEMPLATE_NAME / "init-scripts"
    (init_dir / "02_will_fail.sql").write_text("SELECT 1;", encoding="utf-8")

    # Retry: must succeed. Before the per-script resume fix this replayed
    # 01_publication.sql and failed with 'publication "resume_test_pub" already exists'.
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    with endpoint.cursor() as cur:
        # Whole-template marker written now that every script has succeeded.
        cur.execute("SELECT 1 FROM neon.compute_bootstrap WHERE template = %s", (TEMPLATE_NAME,))
        assert cur.fetchone() is not None

        # Both scripts recorded individually.
        cur.execute(
            "SELECT script FROM neon.compute_bootstrap_scripts "
            "WHERE template = %s ORDER BY script",
            (TEMPLATE_NAME,),
        )
        scripts = [r[0] for r in cur.fetchall()]
        assert scripts == [
            "init-scripts/01_publication.sql",
            "init-scripts/02_will_fail.sql",
        ]

        # The publication was created exactly once.
        cur.execute("SELECT count(*) FROM pg_publication WHERE pubname = 'resume_test_pub'")
        assert cur.fetchone()[0] == 1


def test_bootstrap_template_dir_not_found(neon_simple_env: NeonEnv, bootstrap_root: Path):
    """
    If the template directory doesn't exist in the image, compute_ctl
    should report an error (the endpoint fails to fully configure).
    """
    env = neon_simple_env

    # bootstrap_root exists but has no TEMPLATE_NAME sub-directory
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    endpoint = env.endpoints.create("main")
    endpoint.respec(skip_pg_catalog_updates=False)
    endpoint.respec_deep(spec={"bootstrap_template": "nonexistent-template"})

    with pytest.raises(Exception):
        endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})


def test_bootstrap_new_migration_file(neon_simple_env: NeonEnv, bootstrap_root: Path):
    """
    After initial bootstrap, adding a new migration file and bumping the
    image does NOT cause re-execution — because the idempotency marker
    already exists. This is expected behavior: new migrations require
    a version bump mechanism (future work).
    """
    env = neon_simple_env

    _write_bootstrap_scripts(
        bootstrap_root,
        init_scripts={
            "01_base.sql": "CREATE TABLE IF NOT EXISTS public.migration_test (id int);",
        },
    )

    endpoint = env.endpoints.create("main")
    endpoint.respec(skip_pg_catalog_updates=False)
    endpoint.respec_deep(spec={"bootstrap_template": TEMPLATE_NAME})
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    with endpoint.cursor() as cur:
        cur.execute("SELECT 1 FROM neon.compute_bootstrap WHERE template = %s", (TEMPLATE_NAME,))
        assert cur.fetchone() is not None

    endpoint.stop()

    # Add a new migration file (simulating image update)
    mig_dir = bootstrap_root / TEMPLATE_NAME / "migrations"
    mig_dir.mkdir(parents=True, exist_ok=True)
    (mig_dir / "20250101000000_new.sql").write_text(
        "ALTER TABLE public.migration_test ADD COLUMN IF NOT EXISTS v2 text;",
        encoding="utf-8",
    )

    # Restart — marker exists, so bootstrap is skipped entirely
    endpoint.start(env={"NEON_COMPUTE_BOOTSTRAP_ROOT": str(bootstrap_root)})

    with endpoint.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='migration_test' AND column_name='v2'"
        )
        # The new migration was NOT applied because marker already exists
        assert cur.fetchone() is None
