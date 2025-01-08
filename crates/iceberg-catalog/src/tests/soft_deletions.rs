use crate::api::iceberg::types::Prefix;
use crate::api::iceberg::v1::tables::TablesService;
use crate::api::iceberg::v1::{DataAccess, DropParams, TableParameters};
use crate::api::management::v1::warehouse::{CreateWarehouseResponse, TabularDeleteProfile};
use crate::api::ApiContext;
use crate::catalog::CatalogServer;
use crate::implementations::postgres::{PostgresCatalog, SecretsState};
use crate::service::authz::{AllowAllAuthorizer, Authorizer};
use crate::service::task_queue::TaskQueueConfig;
use crate::service::{State, UserId};
use crate::tests::random_request_metadata;
use crate::WarehouseIdent;
use iceberg::spec::TableMetadata;
use iceberg_ext::catalog::rest::LoadTableResult;
use sqlx::PgPool;

mod test {
    use crate::api::management::v1::warehouse::{
        ListDeletedTabularsQuery, RescheduleSoftDeletionRequest, Service, UndropTabularsRequest,
    };
    use crate::api::management::v1::ApiServer;
    use crate::service::TabularIdentUuid;
    use crate::tests::soft_deletions::{
        load_table, purge_table, setup_drop_test, table_location_exists, DropSetup,
    };
    use crate::tests::{random_request_metadata, spawn_drop_queues};
    use sqlx::PgPool;
    use uuid::Uuid;

    #[sqlx::test]
    async fn test_soft_deletions_are_listable(pool: PgPool) {
        let DropSetup {
            ctx,
            warehouse,
            namespace_name,
            table_name,
            table,
        } = setup_drop_test(
            pool,
            chrono::Duration::milliseconds(125),
            std::time::Duration::from_millis(250),
        )
        .await;

        purge_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name).await;

        let err = load_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name)
            .await
            .unwrap_err();
        assert_eq!(err.error.code, http::StatusCode::NOT_FOUND.as_u16());
        assert!(table_location_exists(&table.metadata).await);

        let r = ApiServer::list_soft_deleted_tabulars(
            warehouse.warehouse_id,
            ListDeletedTabularsQuery {
                namespace_id: Some(Uuid::nil().into()),
                page_token: None,
                page_size: 1,
            },
            ctx.clone(),
            random_request_metadata(),
        )
        .await
        .unwrap();
        assert!(r.tabulars.is_empty());

        let r = ApiServer::list_soft_deleted_tabulars(
            warehouse.warehouse_id,
            ListDeletedTabularsQuery {
                namespace_id: None,
                page_token: None,
                page_size: 10,
            },
            ctx.clone(),
            random_request_metadata(),
        )
        .await
        .unwrap();
        assert!(!r.tabulars.is_empty());
        assert_eq!(r.tabulars.first().unwrap().id, table.metadata.uuid());
        assert!(r.tabulars.first().unwrap().deleted_at < chrono::Utc::now());
        assert!(r.tabulars.first().unwrap().expiration_date > chrono::Utc::now());
    }

    #[sqlx::test]
    async fn test_soft_deleted_tables_are_deleted_after(pool: PgPool) {
        let DropSetup {
            ctx,
            warehouse,
            namespace_name,
            table_name,
            table,
        } = setup_drop_test(
            pool,
            chrono::Duration::milliseconds(125),
            std::time::Duration::from_millis(250),
        )
        .await;

        purge_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name).await;

        spawn_drop_queues(&ctx);

        let err = load_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name)
            .await
            .unwrap_err();
        assert_eq!(err.error.code, http::StatusCode::NOT_FOUND.as_u16());
        assert!(table_location_exists(&table.metadata).await);

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        assert!(!table_location_exists(&table.metadata).await);
    }

    #[sqlx::test]
    async fn test_undrop_makes_table_loadable_again(pool: PgPool) {
        let DropSetup {
            ctx,
            warehouse,
            namespace_name,
            table_name,
            table,
        } = setup_drop_test(
            pool,
            chrono::Duration::seconds(360),
            std::time::Duration::from_secs(10),
        )
        .await;

        purge_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name).await;

        spawn_drop_queues(&ctx);
        let err = load_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name)
            .await
            .unwrap_err();
        assert_eq!(err.error.code, http::StatusCode::NOT_FOUND.as_u16());

        ApiServer::undrop_tabulars(
            random_request_metadata(),
            UndropTabularsRequest {
                targets: vec![TabularIdentUuid::Table(table.metadata.uuid())],
            },
            ctx.clone(),
        )
        .await
        .unwrap();

        let t2 = load_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name)
            .await
            .unwrap();

        assert_eq!(table.metadata.uuid(), t2.metadata.uuid());
    }

    #[sqlx::test]
    async fn test_dropped_table_can_be_dropped_now(pool: PgPool) {
        let DropSetup {
            ctx,
            warehouse,
            namespace_name,
            table_name,
            table,
        } = setup_drop_test(
            pool,
            chrono::Duration::seconds(10),
            std::time::Duration::from_millis(250),
        )
        .await;

        purge_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name).await;

        spawn_drop_queues(&ctx);

        let err = load_table(&ctx, warehouse.warehouse_id, &namespace_name, &table_name)
            .await
            .unwrap_err();
        assert_eq!(err.error.code, http::StatusCode::NOT_FOUND.as_u16());
        assert!(table_location_exists(&table.metadata).await);

        ApiServer::reschedule_soft_deletions(
            random_request_metadata(),
            warehouse.warehouse_id,
            RescheduleSoftDeletionRequest {
                targets: vec![TabularIdentUuid::Table(table.metadata.uuid())],
                reschedule_to: chrono::Utc::now() + chrono::Duration::milliseconds(500),
            },
            ctx.clone(),
        )
        .await
        .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(900)).await;

        assert!(!table_location_exists(&table.metadata).await);
    }
}

struct DropSetup {
    ctx: ApiContext<State<AllowAllAuthorizer, PostgresCatalog, SecretsState>>,
    warehouse: CreateWarehouseResponse,
    namespace_name: String,
    table_name: String,
    table: LoadTableResult,
}

async fn setup_drop_test(
    pool: PgPool,
    expiration_seconds: chrono::Duration,
    poll_interval: std::time::Duration,
) -> DropSetup {
    let prof = crate::tests::test_io_profile();
    let authz = AllowAllAuthorizer::default();
    let (ctx, warehouse) = crate::tests::setup(
        pool.clone(),
        prof,
        None,
        authz,
        TabularDeleteProfile::Soft { expiration_seconds },
        Some(UserId::OIDC("test-user-id".to_string())),
        Some(TaskQueueConfig {
            max_retries: 1,
            max_age: chrono::Duration::seconds(60),
            poll_interval,
        }),
    )
    .await;

    let ns_name = "ns1";
    let tab_name = "tab1".to_string();
    let _ = crate::tests::create_ns(
        ctx.clone(),
        warehouse.warehouse_id.to_string(),
        ns_name.to_string(),
    )
    .await;

    let tab = super::create_table(
        ctx.clone(),
        &warehouse.warehouse_id.to_string(),
        &ns_name,
        &tab_name.clone(),
    )
    .await
    .unwrap();

    DropSetup {
        ctx,
        warehouse,
        namespace_name: ns_name.to_string(),
        table_name: tab_name,
        table: tab,
    }
}

async fn table_location_exists(meta: &TableMetadata) -> bool {
    tokio::fs::try_exists(&meta.location().splitn(2, ":/").collect::<Vec<_>>()[1])
        .await
        .unwrap()
}

async fn load_table<T: Authorizer>(
    ctx: &ApiContext<State<T, PostgresCatalog, SecretsState>>,
    warehouse: WarehouseIdent,
    ns_name: &str,
    tab_name: &str,
) -> crate::api::Result<LoadTableResult> {
    CatalogServer::load_table(
        TableParameters::new(Some(Prefix(warehouse.to_string())), &ns_name, &tab_name),
        DataAccess::none(),
        ctx.clone(),
        random_request_metadata(),
    )
    .await
}

async fn purge_table<T: Authorizer>(
    ctx: &ApiContext<State<T, PostgresCatalog, SecretsState>>,
    warehouse: WarehouseIdent,
    ns_name: &str,
    tab_name: &str,
) {
    CatalogServer::drop_table(
        TableParameters::new(Some(Prefix(warehouse.to_string())), &ns_name, &tab_name),
        DropParams {
            purge_requested: Some(true),
        },
        ctx.clone(),
        random_request_metadata(),
    )
    .await
    .unwrap();
}
