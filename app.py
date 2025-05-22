import sys, os
from animal.pipeline.training_pipeline import TrainPipeline
from animal.logger import logging
from animal.exception import AnimalException
from animal.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from animal.configuration.s3_operations import S3Operation
from animal.entity.config_entity import ModelPusherConfig

# Initialize flask app
app = Flask(__name__)
CORS(app)

# User-provided image class
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training completed successfully."


# Render home page
@app.route("/")
def home():
    return render_template("index.html")


# Download YOLOv11 weights from S3
def download_weights_from_s3():
    """
    Fetch best.pt weights from S3 and place them in yolov11 directory
    """
    try:
        model_pusher_config = ModelPusherConfig()
        s3 = S3Operation()

        destination_dir = os.path.join("yolov11", model_pusher_config.S3_MODEL_KEY_PATH)  # yolov11/best.pt

        if os.path.exists(destination_dir):
            logging.info(f"Model file already exists: {destination_dir}")
        else:
            s3.download_object(
                key=model_pusher_config.S3_MODEL_KEY_PATH,
                bucket_name=model_pusher_config.MODEL_BUCKET_NAME,
                filename=destination_dir
            )
            logging.info(f"Downloaded weights from S3: {destination_dir}")

        return destination_dir

    except Exception as e:
        raise AnimalException(e, sys)


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        # Ensure weights are available locally
        weight_path = download_weights_from_s3()

        # Run YOLOv11 prediction
        os.system(f"yolo task=detect mode=predict model={weight_path} conf=0.25 source={clApp.filename} save=True")

        # Locate output image
        output_img_path = "runs/detect/predict/inputImage.jpg"
        if not os.path.exists(output_img_path):
            raise FileNotFoundError(f"Prediction output not found: {output_img_path}")

        opencodedbase64 = encodeImageIntoBase64(output_img_path)
        result = {"image": opencodedbase64.decode('utf-8')}

        # Clean up
        os.system("rm -rf runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data")
    except KeyError:
        return Response("Key value error: incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)
