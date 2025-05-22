import os
import sys
from animal.configuration.s3_operations import S3Operation
from animal.entity.artifacts_entity import ModelPusherArtifacts, ModelTrainerArtifact
from animal.entity.config_entity import ModelPusherConfig
from animal.exception import AnimalException
from animal.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        s3: S3Operation
    ):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifacts = model_trainer_artifact
        self.s3 = s3

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
        Uploads the YOLOv11 trained model to S3 and returns model pusher artifact.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            trained_model_path = self.model_trainer_artifacts.trained_model_file_path
            logging.info(f"Trained model path: {trained_model_path}")

            if not os.path.exists(trained_model_path):
                raise FileNotFoundError(f"Trained model not found at {trained_model_path}")

            # Upload model to S3
            self.s3.upload_file(
                from_filename=trained_model_path,
                to_filename=self.model_pusher_config.S3_MODEL_KEY_PATH,
                bucket_name=self.model_pusher_config.MODEL_BUCKET_NAME,
                remove=False,
            )
            logging.info(f"Uploaded YOLOv11 model to S3: {self.model_pusher_config.S3_MODEL_KEY_PATH}")

            # Prepare artifact
            model_pusher_artifact = ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.MODEL_BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH,
            )

            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise AnimalException(e, sys) from e
