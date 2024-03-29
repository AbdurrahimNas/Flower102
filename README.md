# Flower102

📌Consider using different model for **faster predictions** and **less memory usage** if deployment would be in a  **mobile or small device**.📌

Image classification for **Flower102 dataset** with **EfficientNet_B2_v2_m** pretrained feature extraction model. It classifies all classes with the whole dataset.


![download](https://github.com/AbdurrahimNas/Flower102/assets/87318891/693637ec-8886-45f9-8e69-7952273a722e)

## How To Use:

### Using Command Prompt:

```
cd ./flower102
```
```
py predict_on_img.py --image_path [IMAGE] --model_path [MODEL]
```

### Using Interface:

```
cd ./flower102/Interface
```
```
py manage.py runserver

```
**After running the server go to**: http://127.0.0.1:8000/FlowerPrediction


**Check out the demo of the app**: https://huggingface.co/spaces/AbdurrahimNas/Flower102


