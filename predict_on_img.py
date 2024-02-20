{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "import warnings\n",
    "from modules.predict import predict\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "parser = argparse.ArgumentParser(description=\"Predicts on a single image.\")\n",
    "parser.add_argument(\"--image_path\", \"-p\", type=str,\n",
    "                    help=\"path of the image that would be predicted upon.\" )\n",
    "parser.add_argument(\"--model_path\", \"-mp\", type=str,\n",
    "                    help=\"path of the model that would predict on a single image.\")\n",
    "\n",
    "args = parser.parse_args()\n",
    "print(f\"Predicted Label: {predict(args.image_path, args.model_path)}\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
