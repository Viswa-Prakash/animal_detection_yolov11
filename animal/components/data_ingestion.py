import os
import sys
import zipfile
from animal.logger import logging
from animal.exception import AnimalException
from animal.entity.config_entity import DataIngestionConfig
from animal.entity.artifacts_entity import DataIngestionArtifact
from animal.configuration.s3_operations import S3Operation
from animal.constant.training_pipeline import DATA_BUCKET_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3 = S3Operation()
        except Exception as e:
            raise AnimalException(e, sys)

    def download_data(self) -> str:
        """
        Downloads zip dataset from S3 to local directory.
        """
        try:
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            logging.info(f"Downloading data from S3 into directory: {zip_download_dir}")

            zip_file_path = os.path.join(zip_download_dir, self.data_ingestion_config.S3_DATA_NAME)

            self.s3.download_object(
                key=self.data_ingestion_config.S3_DATA_NAME,
                bucket_name=DATA_BUCKET_NAME,
                filename=zip_file_path,
            )

            logging.info(f"Data successfully downloaded to {zip_file_path}")
            return zip_file_path

        except Exception as e:
            raise AnimalException(e, sys)

    def extract_zip_file(self, zip_file_path: str) -> str:
        """
        Extracts the zip file into the feature store directory.
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)

            logging.info(f"Extracting {zip_file_path} to {feature_store_path}")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)

            logging.info("Extraction complete.")
            return feature_store_path

        except Exception as e:
            raise AnimalException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates data ingestion process.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path=zip_file_path,
                feature_store_path=feature_store_path
            )

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise AnimalException(e, sys)
