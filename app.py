import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image

# Thiết lập thư mục cache cho Transformers
os.environ["TRANSFORMERS_CACHE"] = "./photoanalysis/transformers_cache"

# Khởi tạo processor và model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Hàm tạo chú thích ảnh
def generate_caption(raw_image: Image) -> str:
    inputs = processor(raw_image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Tạo Gradio interface
image = gr.Image(label="Ảnh Đầu vào!")
caption = gr.Textbox(label="Chú thích ảnh", placeholder="Chú thích ảnh sẽ được đưa ra tại đây")
demo = gr.Interface(fn=generate_caption, inputs=[image], outputs=[caption])

# Khởi động ứng dụng
if __name__ == "__main__":
    demo.launch()
