import os
import sys
import yaml
from ultralytics import YOLO
from six.moves import urllib
from animal.logger import logging
from animal.exception import AnimalException
from animal.constant.training_pipeline import *
from animal.entity.config_entity import ModelTrainerConfig
from animal.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Unzip dataset
            logging.info("Unzipping data...")
            os.system(f"unzip {DATA_INGESTION_S3_DATA_NAME}")
            os.system(f"rm {DATA_INGESTION_S3_DATA_NAME}")

            #Prepare image path in txt file
            train_img_path = os.path.join(os.getcwd(),"images","train")
            val_img_path = os.path.join(os.getcwd(),"images","val")

            #Training images
            with open('train.txt', "a+") as f:
                img_list = os.listdir(train_img_path)
                for img in img_list:
                    f.write(os.path.join(train_img_path,img+'\n'))
                print("Done Training images")


            # Validation Image
            with open('val.txt', "a+") as f:
                img_list = os.listdir(val_img_path)
                for img in img_list:
                    f.write(os.path.join(val_img_path,img+'\n'))
                print("Done Validation Image")


            # Download YOLOv11 weights if provided via URL
            url = self.model_trainer_config.weight_name
            weight_filename = os.path.basename(url)
            weight_path = os.path.join("weights", weight_filename)
            os.makedirs("weights", exist_ok=True)

            if not os.path.exists(weight_path):
                logging.info(f"Downloading YOLOv11 weight from {url}")
                urllib.request.urlretrieve(url, weight_path)

            # Load YOLOv11 model
            logging.info("Initializing YOLOv11 model")
            model = YOLO(weight_path)

            # Train the model
            logging.info("Starting training with YOLOv11")
            model.train(
                data=DATA_YAML_PATH,
                epochs=self.model_trainer_config.no_epochs,
                batch=self.model_trainer_config.batch_size,
                #imgsz=self.model_trainer_config.image_size,
                project=self.model_trainer_config.model_trainer_dir,
                name="yolov11_training",
                exist_ok=True
            )

            # Path to best model after training
            best_model_path = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "yolov11_training",
                "weights",
                "best.pt"
            )

            # Save model artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=best_model_path
            )

            logging.info("Model training complete.")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise AnimalException(e, sys)
