alter table task
    add column state      jsonb,
    add column tabular_id uuid references tabular (tabular_id) on delete set null
-- TODO: cancellable
-- TODO: bad-state to mark deserialization issues
;

create index if not exists task_tabular_id_idx
    on task (tabular_id);

update task
set state      = jsonb_build_object(
        'tabular_id', te.tabular_id,
        'typ', te.typ,
        'deletion_kind', te.deletion_kind),
    tabular_id = te.tabular_id
from tabular_expirations te
where te.task_id = task.task_id;

update task
set state      = jsonb_build_object(
        'tabular_id', tp.tabular_id,
    -- TODO: check what happens here and if we need a cast to text
        'typ', tp.typ,
        'tabular_location', tp.tabular_location),
    tabular_id = tp.tabular_id
from tabular_purges tp
where tp.task_id = task.task_id;

drop table tabular_expirations;
drop table tabular_purges;

alter table task
    alter column state set not null;

create table task_config
(
    warehouse_id uuid references warehouse (warehouse_id) on delete cascade not null,
    queue_name   text                                                       not null,
    config       jsonb                                                      not null,
    primary key (warehouse_id, queue_name)
);
