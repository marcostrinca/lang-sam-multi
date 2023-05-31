
import os
import warnings

import gradio as gr
import numpy as np
from PIL import Image

from lang_sam import LangSAM
from lang_sam import SAM_MODELS
from lang_sam.utils import draw_image
from lang_sam.utils import load_image

warnings.filterwarnings("ignore")
ready = False
g_sam_type = "vit_h"
text_prompt = "foreground"
model = LangSAM()

gr_inputs = [
        gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h"),
        gr.Slider(0, 1, value=0.3, label="Box threshold"),
        gr.Slider(0, 1, value=0.25, label="Text threshold"),
        gr.Image(type="filepath", label='Image'),
        #gr.inputs.File(file_count="multiple", label="Image list"),
        gr.Textbox(lines=1, label="Text Prompt"),
    ]
    
def predict(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)

    image_pil = load_image(image_path)

    #image_pil2 = load_image(image_list[0])
    #print(image_list[0].name)

    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt, box_threshold, text_threshold)
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    return image

gr_outputs = [gr.outputs.Image(type="pil", label="Output Image")]

gr.Interface(fn=predict,
             inputs=gr_inputs,
             outputs=gr_outputs).launch(server_name="0.0.0.0")
