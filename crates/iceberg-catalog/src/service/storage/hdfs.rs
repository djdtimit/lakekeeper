// use std::{
//     collections::HashMap,
//     str::FromStr,
//     sync::{Arc, OnceLock},
// };

use iceberg_ext::configs::{table::TableProperties, Location, ParseFromStr};
use serde::{Deserialize, Serialize};

use crate::{
    api::iceberg::v1::DataAccess,
    service::storage::{
        error::TableConfigError, StoragePermissions, StorageType, TableConfig, ValidationError,
    },
};

#[derive(Debug, Eq, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub struct HdfsProfile {
    pub url: String,
    pub prefix: String,
}

impl HdfsProfile {
    pub fn generate_table_config(
        &self,
        _data_access: &DataAccess,
        _table_location: &Location,
        _storage_permissions: StoragePermissions,
    ) -> Result<TableConfig, TableConfigError> {
        Ok(TableConfig {
            creds: TableProperties::default(),
            config: TableProperties::default(),
        })
    }

    pub fn normalize(&mut self) -> Result<(), ValidationError> {
        Ok(())
    }

    pub fn base_location(&self) -> Result<Location, ValidationError> {
        Location::parse_value(&format!("hdfs://{}", self.prefix.as_str())).map_err(|e| {
            ValidationError::InvalidLocation {
                source: Some(Box::new(e)),
                reason: "Failed to create location for storage profile.".to_string(),
                storage_type: StorageType::Adls,
                location: self.prefix.clone(),
            }
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, utoipa::ToSchema, PartialEq)]
pub struct HdfsCredential {}

#[cfg(test)]
pub(crate) mod test {
    use std::collections::HashSet;

    use hdfs_native::minidfs::MiniDfs;
    use serial_test::serial;

    use super::*;
    use crate::service::storage::{StorageCredential, StorageProfile};

    #[sqlx::test]
    #[serial]
    async fn test_can_validate() {
        let (minidfs, hdfs_profile, cred) = test_profile();
        let prof = StorageProfile::Hdfs(hdfs_profile);
        prof.validate_access(None, None).await.unwrap()
    }

    pub(crate) fn test_profile() -> (MiniDfs, HdfsProfile, StorageCredential) {
        let minidfs = MiniDfs::with_features(&HashSet::default());

        let hdfs_profile = HdfsProfile {
            url: minidfs.url.clone(),
            prefix: "hdfs:///user/hdfs".to_string(),
        };

        (
            minidfs,
            hdfs_profile,
            StorageCredential::Hdfs(HdfsCredential {}),
        )
    }
}
