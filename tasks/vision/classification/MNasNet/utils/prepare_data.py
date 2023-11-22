import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image
import argparse

h, w = 224, 224

def prepare_image(image):
    image = np.array(image['image'])
    image = Image.fromarray(image)
    image = image.resize((h, w))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def inference(interpreter, tensor):
    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()

    if tensor.shape[-1] != 3:
        return

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[1]["index"])

    return pred_y.argmax()

if __name__ == "__main__":
    MODEL_NAME = "mnasnet-a1-075"
    model_path = "keras_model/" + MODEL_NAME + "/saved_model"

    converter = tf.lite.TFLiteConverter.from_saved_model(
        model_path, signature_keys=["classify"]
    )

    converter.input_shape = (1, h, w, 3)
    tflite_quant_model = converter.convert()

    with open(MODEL_NAME + "_float32.tflite", "wb") as f:
        f.write(tflite_quant_model)

    correct_imgs = np.empty((1, h, w, 3), dtype=np.uint8)

    ds, ds_info = tfds.load('imagenet_v2/topimages', split='test', with_info=True)
    for file in tqdm(ds.take(1000)):
        y_true = file['label']
        y_true = y_true.numpy()
        tensor = prepare_image(file)

        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
        pred_y = inference(interpreter, tensor)

        if not pred_y:
            continue

        if y_true == pred_y:
            correct_imgs = np.append(correct_imgs, tensor, axis=0)

    np.save('quantization_data.npy', correct_imgs)