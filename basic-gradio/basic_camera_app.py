import gradio as gr


def process_image(image):
    return image


iface = gr.Interface(
    fn=process_image, inputs=gr.Image(sources="webcam"), outputs=gr.Image()
)
iface.launch()
