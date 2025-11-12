# -*- coding: utf-8 -*-
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
# import matplotlib.pyplot as plt
from scripts.base.utils import transform
from scripts.base.constants import TOP, HEATMAP_FOLDER


def use_base_model(device, model, classes, best_model_path, image_path, model_name="", gradcam=False):
    debug_file = False
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()   
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    
    # === Grad-CAM setup ===
    if gradcam:
        target_layer = model.backbone.layer4[-1]
        activations, gradients = [], []
    
        def save_activation(_, __, output):
            activations.append(output)
    
        def save_gradient(_, grad_input, grad_output):
            gradients.append(grad_output[0])
    
        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)
        
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    pred_idx = probs.argmax()
    pred_class = classes[pred_idx]
    result = {cls: round(p * 100, 2) for cls, p in zip(classes, probs)}
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    top_results = dict(sorted_result[:TOP])
    pred_class, pred_class_procent = sorted_result[0]
    print(f"\n Image: {image_path}")
    print(f"Predicted Boat Model: {pred_class}")
    print(f"Class probabilities (top {str(TOP)}):")
    # Show top:
    for cls, prob in top_results.items():
        print(f"   {cls:<25} {prob:>6.2f}%")
    # === Grad-CAM visualization ===
    if gradcam:
        model.zero_grad()
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        output = outputs[0, pred_idx]
        output.backward()

        grads = gradients[0].detach()
        acts = activations[0].detach()

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / cam.max()

        # === Overlay heatmap ===
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (img.size[0], img.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_np = np.array(img)
        overlay = np.uint8(np.clip(0.6 * img_np + 0.4 * heatmap, 0, 255))

        # === Add class probabilities to overlay ===
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)[:TOP]
        for i, (cls, prob) in enumerate(sorted_results):
            text = f"{cls.upper()}: {prob:.2f}%"
            cv2.putText(
                overlay,
                text,
                (20, 40 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        os.makedirs(HEATMAP_FOLDER, exist_ok=True)
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        new_base_name = uuid.uuid4().hex
        debug_file = f"{new_base_name}{ext}"
        output_path = os.path.join(HEATMAP_FOLDER, debug_file)
        Image.fromarray(overlay).save(output_path)

        print(f"\n✅ Grad-CAM saved: {output_path}")

    return pred_class, pred_class_procent, top_results, debug_file