# Closes the TOCTOU window between the runner's `_has_running_run` check and the
# row insert via a partial unique index. Django's `UniqueConstraint(..., condition=...)`
# normally compiles to a non-concurrent `CREATE UNIQUE INDEX`, which locks the table
# while the index builds. We use `SeparateDatabaseAndState` so Django's model state
# still tracks the constraint while the actual DDL stays online via
# `CREATE UNIQUE INDEX CONCURRENTLY`. Same pattern as the rest of the codebase's
# safe non-blocking index migrations.

from django.db import migrations, models


class Migration(migrations.Migration):
    # Required for CONCURRENTLY — Postgres rejects it inside a transaction.
    atomic = False

    dependencies = [
        ("posthog", "1146_subscription_enabled"),
        ("signals", "0022_signalagentconfig_runs_per_tick"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.AddConstraint(
                    model_name="signalagentrun",
                    constraint=models.UniqueConstraint(
                        condition=models.Q(("status", "running")),
                        fields=("team", "skill_name"),
                        name="signal_agent_run_one_running_per_team_skill",
                    ),
                ),
            ],
            database_operations=[
                migrations.RunSQL(
                    sql=(
                        "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
                        "signal_agent_run_one_running_per_team_skill "
                        'ON "signals_signalagentrun" (team_id, skill_name) '
                        "WHERE status = 'running'"
                    ),
                    reverse_sql=("DROP INDEX CONCURRENTLY IF EXISTS signal_agent_run_one_running_per_team_skill"),
                ),
            ],
        ),
    ]
