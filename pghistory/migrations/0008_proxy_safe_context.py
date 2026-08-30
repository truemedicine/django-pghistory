from django.db import migrations

from pghistory.models import Context


def install_pgh_set_context_func(apps, schema_editor):
    Context.install_pgh_set_context_func(using=schema_editor.connection.alias)


class Migration(migrations.Migration):
    dependencies = [("pghistory", "0007_auto_20250421_0444")]

    operations = [
        migrations.RunPython(
            install_pgh_set_context_func,
            reverse_code=migrations.RunPython.noop,
        )
    ]
