import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm

def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Use broadcasting instead of modifying tensor
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize between 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_gradcam_overlay(img_path, heatmap, cam_path="static/gradcam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap)
    colored_heatmap = np.uint8(255 * colored_heatmap[..., :3])
    superimposed_img = cv2.addWeighted(img, 1 - alpha, colored_heatmap, alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path
