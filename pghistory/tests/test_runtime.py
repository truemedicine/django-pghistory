import concurrent.futures
import uuid

import pytest
from django.db import ProgrammingError, connection, connections, transaction
from django.db.models import F
from django.utils import timezone

import pghistory.runtime
import pghistory.tests.models as test_models
import pghistory.utils


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("create index concurrently", True),
        ("create index", True),
        ("select * from auth_user", True),
        ("vacuum table", True),
        ("analyze table", True),
        ("checkpoint table", True),
        ("discard all", True),
        ("load extension", True),
        ("cluster", True),
        ("update auth_user set id= %s where id = %s", False),
        (b"create index concurrently", True),
        (b"select * from auth_user", True),
        (b"update auth_user set id= %s where id = %s", False),
        ("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", True),
        ("set transaction isolation level read committed", True),
        ("SET TRANSACTION READ ONLY", True),
        ("set transaction read write", True),
        ("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE", True),
        ("BEGIN", True),
        ("begin transaction", True),
        ("START TRANSACTION", True),
        ("start transaction isolation level repeatable read", True),
        ("SELECT * FROM table", True),
        ("INSERT INTO table VALUES (1)", False),
        ("UPDATE table SET col = 1", False),
        ("DELETE FROM table", False),
        ("SET search_path = public", False),
        (b"SET TRANSACTION ISOLATION LEVEL REPEATABLE READ", True),
        (b"SELECT * FROM table", True),
    ],
)
def test_is_ignored_statement(statement, expected):
    assert pghistory.runtime._is_ignored_statement(statement) == expected


@pytest.mark.skipif(
    pghistory.utils.psycopg_maj_version == 3, reason="Psycopg2 preserves entire query"
)
@pytest.mark.django_db
@pytest.mark.parametrize("context_setter", ["direct", "function"])
@pytest.mark.parametrize(
    "sql, params",
    [
        ("update auth_user set id= %s where id = %s", (1, 1)),
        ("update auth_user set id= %(id1)s where id = %(id2)s", {"id1": 1, "id2": 1}),
        (b"update auth_user set id= %s where id = %s", (1, 1)),
        (b"update auth_user set id= %(id1)s where id = %(id2)s", {"id1": 1, "id2": 1}),
    ],
)
def test_inject_history_context(settings, mocker, context_setter, sql, params):
    context_id = uuid.UUID(int=0)
    mocker.patch("uuid.uuid4", return_value=context_id, autospec=True)
    settings.DEBUG = True
    settings.PGHISTORY_CONTEXT_SETTER = context_setter
    if context_setter == "function":
        expected_sql = (
            f"SELECT _pgh_set_context('{context_id}'::uuid, '{{\"hello\": \"world\"}}'::jsonb);"  # noqa
        )
    else:
        expected_sql = f"SELECT set_config('pghistory.context_id', '{context_id}', true), set_config('pghistory.context_metadata', '{{\"hello\": \"world\"}}', true);"  # noqa

    with pghistory.context(hello="world"):
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            query = connection.queries[-1]
            assert query["sql"].startswith(expected_sql)


@pytest.mark.django_db(transaction=True)
def test_context_function_preserves_context_contract(settings):
    settings.PGHISTORY_CONTEXT_SETTER = "function"
    with transaction.atomic():
        with pghistory.context(request_id="request-1") as tracked_context:
            tracked = test_models.SnapshotModel.objects.create(
                dt_field=timezone.now(), int_field=0
            )
            with pghistory.context(user=17) as nested_context:
                tracked.int_field = 1
                tracked.save(update_fields=["int_field"])

            tracked.int_field = 2
            tracked.save(update_fields=["int_field"])

    assert nested_context.id == tracked_context.id
    assert pghistory.models.Context.objects.get(id=tracked_context.id).metadata == {
        "request_id": "request-1",
        "user": 17,
    }
    assert set(
        test_models.SnapshotModelSnapshot.objects.filter(pgh_obj_id=tracked.id).values_list(
            "pgh_context_id", flat=True
        )
    ) == {tracked_context.id}

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                NULLIF(current_setting('pghistory.context_id', true), ''),
                NULLIF(current_setting('pghistory.context_metadata', true), '')
            """
        )
        assert cursor.fetchone() == (None, None)


@pytest.mark.django_db(transaction=True)
def test_context_function_rollback_isolation(settings):
    settings.PGHISTORY_CONTEXT_SETTER = "function"
    tracked = test_models.SnapshotModel.objects.create(dt_field=timezone.now(), int_field=0)
    original_event_count = test_models.SnapshotModelSnapshot.objects.count()

    class Rollback(Exception):
        pass

    with pytest.raises(Rollback):
        with transaction.atomic():
            with pghistory.context(rolled_back=True) as rolled_back_context:
                tracked.int_field = 1
                tracked.save(update_fields=["int_field"])
                raise Rollback

    tracked.refresh_from_db()
    assert tracked.int_field == 0
    assert test_models.SnapshotModelSnapshot.objects.count() == original_event_count
    assert not pghistory.models.Context.objects.filter(id=rolled_back_context.id).exists()


@pytest.mark.django_db(transaction=True)
def test_context_function_isolates_concurrent_connections(settings):
    settings.PGHISTORY_CONTEXT_SETTER = "function"
    tracked_models = [
        test_models.SnapshotModel.objects.create(dt_field=timezone.now(), int_field=worker)
        for worker in range(8)
    ]

    def update_with_context(worker_and_id):
        worker, tracked_id = worker_and_id
        try:
            with pghistory.context(worker=worker) as tracked_context:
                test_models.SnapshotModel.objects.filter(id=tracked_id).update(
                    int_field=F("int_field") + 1
                )
            return tracked_context.id
        finally:
            connections["default"].close()

    inputs = [(worker, tracked.id) for worker, tracked in enumerate(tracked_models)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        context_ids = list(executor.map(update_with_context, inputs))

    assert len(set(context_ids)) == 8
    events = test_models.SnapshotModelSnapshot.objects.filter(
        pgh_label="snapshot_update",
        pgh_obj_id__in=[tracked.id for tracked in tracked_models],
    ).select_related("pgh_context")
    assert {event.pgh_obj_id: event.pgh_context.metadata for event in events} == {
        tracked.id: {"worker": worker} for worker, tracked in enumerate(tracked_models)
    }


@pytest.mark.django_db(transaction=True)
def test_context_function_missing_fails_loudly(settings):
    settings.PGHISTORY_CONTEXT_SETTER = "function"
    tracked = test_models.SnapshotModel.objects.create(dt_field=timezone.now(), int_field=0)
    with connection.cursor() as cursor:
        cursor.execute(
            "ALTER FUNCTION _pgh_set_context(UUID, JSONB) RENAME TO _pgh_set_context_disabled"
        )

    try:
        with pytest.raises(ProgrammingError):
            with transaction.atomic():
                with pghistory.context():
                    tracked.int_field = 1
                    tracked.save(update_fields=["int_field"])
    finally:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER FUNCTION _pgh_set_context_disabled(UUID, JSONB) RENAME TO _pgh_set_context"
            )
