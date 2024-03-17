import argparse
import warnings
from modules.predict import predict
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Predicts on a single image.")
parser.add_argument("--image_path", "-p", type=str,
                    help="path of the image that would be predicted upon." )
parser.add_argument("--model_path", "-mp", type=str,
                    help="path of the model that would predict on a single image.")

args = parser.parse_args()
prediction = predict(args.image_path, args.model_path)
print(f"Predicted Label: {prediction}")
