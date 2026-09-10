"""The optional backend hook must preserve the caller's execution contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from django.db import connection

import pghistory
from pghistory import runtime


@pytest.fixture
def cursor():
    return SimpleNamespace(
        name=None,
        connection=SimpleNamespace(
            info=SimpleNamespace(transaction_status=0), get_transaction_status=lambda: 0
        ),
    )


@pytest.mark.parametrize("setter", ["direct", "function"])
@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("kind", ["tuple", "mapping", "none", "many"])
def test_handoff_preserves_sql_and_parameters(settings, cursor, setter, as_bytes, kind):
    settings.PGHISTORY_CONTEXT_SETTER = setter
    sql = (
        "UPDATE example SET value = %(value)s"
        if kind == "mapping"
        else "UPDATE example SET value = %s"
    )
    sql = sql.encode() if as_bytes else sql
    rows = []

    def lazy_rows():
        rows.append("consumed")
        yield (42,)

    params = {
        "tuple": (42,),
        "mapping": {"value": 42, "pghistory__context_id": "caller-owned"},
        "none": None,
        "many": lazy_rows(),
    }[kind]
    prepend = Mock()
    execute = Mock()
    context = {"cursor": cursor, "pg_prepend_sql": prepend}
    with pghistory.context(hello="world") as tracked:
        result = runtime._inject_history_context(execute, sql, params, kind == "many", context)

    prefix = (
        "SELECT _pgh_set_context(%s::uuid, %s::jsonb)"
        if setter == "function"
        else "SELECT set_config('pghistory.context_id', %s, true), "
        "set_config('pghistory.context_metadata', %s, true)"
    )
    prepend.assert_called_once_with(prefix, (str(tracked.id), '{"hello": "world"}'))
    execute.assert_called_once_with(sql, params, kind == "many", context)
    assert execute.call_args.args[0] is sql
    assert execute.call_args.args[1] is params
    assert rows == []
    if kind == "mapping":
        assert params == {"value": 42, "pghistory__context_id": "caller-owned"}
    assert result is execute.return_value
    result.nextset.assert_not_called()


@pytest.mark.parametrize("skip", ["select", "named", "errored"])
def test_handoff_preserves_injection_exclusions(cursor, skip):
    sql = "SELECT 1" if skip == "select" else "UPDATE example SET value = 1"
    if skip == "named":
        cursor.name = "named"
    if skip == "errored":
        cursor.connection.info.transaction_status = 3
        cursor.connection.get_transaction_status = lambda: 3
    prepend = Mock()
    execute = Mock()
    context = {"cursor": cursor, "pg_prepend_sql": prepend}
    with pghistory.context():
        assert (
            runtime._inject_history_context(execute, sql, None, False, context)
            is execute.return_value
        )
    prepend.assert_not_called()
    execute.assert_called_once_with(sql, None, False, context)
    execute.return_value.nextset.assert_not_called()


@pytest.mark.parametrize("setter", ["direct", "function"])
@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_without_hook_retains_combined_execution(settings, cursor, setter, as_bytes, named):
    settings.PGHISTORY_CONTEXT_SETTER = setter
    sql = "UPDATE example SET value = %(value)s" if named else "UPDATE example SET value = %s"
    params = {"value": 42} if named else (42,)
    execute = Mock()
    execute.return_value.nextset.return_value = None
    with pghistory.context() as tracked:
        runtime._inject_history_context(
            execute, sql.encode() if as_bytes else sql, params, False, {"cursor": cursor}
        )
    sent_sql, sent_params, many, _ = execute.call_args.args
    assert isinstance(sent_sql, bytes) == as_bytes
    sent_sql = sent_sql.decode() if as_bytes else sent_sql
    assert sent_sql.endswith("; " + sql)
    assert sent_sql.startswith(
        "SELECT _pgh_set_context(" if setter == "function" else "SELECT set_config("
    )
    assert many is False
    if named:
        assert sent_params == {
            "value": 42,
            "pghistory__context_id": str(tracked.id),
            "pghistory__context_metadata": "{}",
        }
    else:
        assert sent_params == (str(tracked.id), "{}", 42)
    if runtime.utils.psycopg_maj_version == 3:
        execute.return_value.nextset.assert_called_once_with()


def test_hook_failure_stops_execution_and_public_context_unwinds(cursor):
    execute = Mock()
    prepend = Mock(side_effect=ValueError("rejected prefix"))
    wrappers = list(connection.execute_wrappers)
    with pytest.raises(ValueError, match="rejected prefix"), pghistory.context():
        runtime._inject_history_context(
            execute,
            "UPDATE example SET value = 1",
            None,
            False,
            {"cursor": cursor, "pg_prepend_sql": prepend},
        )
    execute.assert_not_called()
    assert connection.execute_wrappers == wrappers


@pytest.fixture
def prefix_backend(monkeypatch):
    """A test backend using only public Django and psycopg execution APIs."""
    if runtime.utils.psycopg_maj_version != 3:
        pytest.skip("Server-side binding requires psycopg 3")
    import contextlib

    import psycopg
    from django.db import connection

    connection.ensure_connection()
    raw = connection.connection
    monkeypatch.setattr(raw, "cursor_factory", psycopg.Cursor)
    observed = []

    def collect(execute, sql, params, many, context):
        prefixes = []
        context["pg_prepend_sql"] = lambda query, values: prefixes.insert(0, (query, values))
        context["test_prefixes"] = prefixes
        return execute(sql, params, many, context)

    def dispatch(execute, sql, params, many, context):
        prefixes = context["test_prefixes"]
        observed.append((prefixes.copy(), sql, params, many))
        with (
            connection.wrap_database_errors,
            raw.transaction(),
            contextlib.ExitStack() as cursors,
            raw.pipeline(),
        ):
            for prefix_sql, prefix_params in prefixes:
                cursors.enter_context(raw.cursor()).execute(prefix_sql, prefix_params)
            return execute(sql, params, many, context)

    return collect, dispatch, observed


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("setter", ["direct", "function"])
def test_public_wrappers_with_server_binding(settings, prefix_backend, setter):
    from django.db import IntegrityError

    settings.PGHISTORY_CONTEXT_SETTER = setter
    collect, dispatch, observed = prefix_backend
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE handoff_history "
            "(value int PRIMARY KEY, context_id uuid, metadata jsonb)"
        )
        try:
            sql = (
                "INSERT INTO handoff_history VALUES (%(value)s, "
                "current_setting('pghistory.context_id')::uuid, "
                "current_setting('pghistory.context_metadata')::jsonb)"
            )
            with (
                connection.execute_wrapper(collect),
                pghistory.context(request="handoff") as tracked,
                connection.execute_wrapper(dispatch),
            ):
                params = {"value": 1}
                cursor.execute(sql + " RETURNING value", params)
                assert cursor.fetchone() == (1,)
                assert cursor.nextset() is None
                assert params == {"value": 1}
                cursor.executemany(sql, ({"value": i} for i in (2, 3)))
                assert cursor.rowcount == 2
                with pytest.raises(IntegrityError):
                    cursor.executemany(sql, ({"value": i} for i in (4, 1)))
            assert all(len(prefixes) == 1 for prefixes, *_ in observed)
            with connection.cursor() as check:
                check.execute(
                    "SELECT value, context_id, metadata::text FROM handoff_history ORDER BY value"
                )
                assert check.fetchall() == [
                    (i, tracked.id, '{"request": "handoff"}') for i in (1, 2, 3)
                ]
                check.execute("SELECT NULLIF(current_setting('pghistory.context_id', true), '')")
                assert check.fetchone() == (None,)
        finally:
            cursor.execute("DROP TABLE handoff_history")
