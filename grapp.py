
import os
import warnings
import shutil

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
masksOut = []
imgNames = []
    
def predict(sam_type, box_threshold, text_threshold, image_path, text_prompt):
    print("Predicting... ", sam_type, box_threshold, text_threshold, image_path, text_prompt)
    masksOut = []
    imgNames = []

    for i in image_path:
        im = load_image(i.name)
        masks, boxes, phrases, logits = model.predict(im, text_prompt, box_threshold, text_threshold)
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
        image_array = np.asarray(im)
        image = draw_image(image_array, masks, boxes, labels)
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        masksOut.append(image)
        imgNames.append(i.name)

    return masksOut, imgNames

def saveMasks(masks, names):
    list_1 = names.split(",")
        # print ("extenxtion: ", imgName.rsplit('.', maxsplit=1)[1])

    for i in range(len(masks)):
        list_2 = list_1[i].split("/")
        imgFullName = list_2[len(list_2)-1]
        imgName = imgFullName.rsplit('.', maxsplit=1)[0]
        target=f"./flagged/{imgName}.png"

        print(masks[i]['name'])
        print(target)

        shutil.copy(masks[i]['name'], target)

with gr.Blocks() as demo:
    with gr.Row(): 
        with gr.Column():
            dropdown = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM model", value="vit_h")
            slider1 = gr.Slider(0, 1, value=0.35, label="Box threshold")
            slider2 = gr.Slider(0, 1, value=0.25, label="Text threshold")
            # imageIn = gr.Image(type="filepath", label='Image')
            images = gr.inputs.File(file_count="multiple", label="Lista")
            textPrompt = gr.Textbox(lines=1, label="O que você quer recortar?")
            send_button = gr.Button("Gerar máscaras")
        
        with gr.Column():
            gallery = gr.Gallery(label="Máscaras geradas", show_label=False, elem_id="gallery").style(columns=[3], object_fit="contain", height="auto")
            fileNames = gr.Textbox()

            save = gr.Button("Salvar máscaras")

    send_button.click(predict, inputs=[dropdown, slider1, slider2, images, textPrompt], outputs=[gallery, fileNames])
    save.click(saveMasks, inputs=[gallery, fileNames])


if __name__ == "__main__":
    demo.launch(allow_flagging="manual") 
