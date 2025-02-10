create table endpoint_stats
(
    warehouse_id uuid references warehouse (warehouse_id) on delete cascade,
    --  warehouse_name text collate "case_insensitive" references warehouse (warehouse_name) on update cascade on delete cascade,
    project_id   uuid references project (project_id) on delete cascade,
    matched_path text        not null,
    method       text        not null,
    status_code  int         not null,
    count        bigint      not null default 0,
    -- we keep stats in hourly intervals, every hour we create a new row,
    valid_until  timestamptz not null,
    primary key (project_id, warehouse_id, matched_path, method, status_code, valid_until)
) PARTITION BY RANGE (valid_until);



select trigger_updated_at('endpoint_stats');
call add_time_columns('endpoint_stats');