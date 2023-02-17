import streamlit as st
import base64
from PIL import Image
# from st_keyup import st_keyup
from diffusers import StableDiffusionPipeline
from st_keyup import st_keyup
import torch
# import os
from diffusers import DiffusionPipeline,StableDiffusionOnnxPipeline
import time
import json

# url="http://127.0.0.2:31766/parse"
#
# headers={
#     "Content-Type":"application/json"
# }

@st.cache(allow_output_mutation=True)
def load_model():
    # model_id="runwayml/stable-diffusion-v1-5"
    pipe=DiffusionPipeline.from_pretrained("prompthero/openjourney-v2",torch_dtype=torch.float16).to("cuda")
    # pipe=StableDiffusionOnnxPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5",
    #     revision="onnx",
    #     provider="CUDAExecutionProvider",
    #     #torch_dtype=torch.float16
    # )
    # pipe = StableDiffusionOnnxPipeline.from_pretrained(
    #     "/home/qian/",
    #     revision="onnx",
    #     provider="CUDAExecutionProvider",
    #     # torch_dtype=torch.float16
    # ).to("cuda")
    #
    # pipe=StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     revision="onnx",
    #     provider="CUDAExecutionProvider",
    #     # torch_dtype=torch.float16
    #
    # ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

pipe=load_model()

st.title("AIGC\n")
st.subheader("输入")

image_path=os.listdir("asaa")
with st.sidebar:
    st.header("example")
    st.code("cat")
    st.header("aa")
    st.text("asd")
    # for ip in image_path:
    #     st.image(ip)
    #     st.text("\n\n\n\n")


# model_id="路径"
# pipe=StableDiffusionPipeline.from_pretrained()
input_query=st_keyup("Enter value",value="boy")
options=st.multiselect("选择词语",
                       ("8k resolution","detailed","3D","4K detailed post processing","trending on artstation","digital painting","artstation","beautiful"),
                       default=("8k resolution"))
print(options)
negative_prompt=st_keyup("Enter a negative prompt",value="grotesque,insightly,misshapen,deformed,mangled,awkard,distorted,twisted,contorted,twisted,contorted,misshapen,lopsided,"
                                                     "malformed,asymmetrical,irregular,unnatural,botched,mangled,mutilated,disfigured,ugly,offensive,respulsive,revolting,ghastly,hideous,unappealing,terrible,awful,frightful,odious,loathsome,revolting,obnoxious,detestable,hateful,repugnant,sickening,vile,abhorrent,contemptible,execrable,repellent,disgusting,revolting,loathsome,distasteful,abominable,ugly,tiling,poorly drawn hands,poorly drawn feet,poorly drawn face,out of frame,extra limbs,disfigured,deformed,body out of frame,blurry,bad anatomy,blurred,watermark,grainy,signature,cut off,draft,amateur,mutuple,gross,weird,uneven,furnishing,decorating,decoration,furnture,text,poor,basic,worst,juvenile,unprofessional,failture,crayon,oil,label,thousand hands")
input_query+=","+",".join([i for i in options])
print(input_query)

a=st.button("确认",key="enter")
print(a)
if a:
    with st.spinner("Please wait a moment..."):

        start_time=time.time()
        with torch.inference_mode():
            image=pipe(input_query,negative_prompt=-negative_prompt).images[0]
            print("latency",time.time()-start_time)
            st.image(image)



    # if a:
    #     stat_time=time.time()
    #     with torch.inference_mode():
            # payload=json.dump({
            #     "query":input_query,
            #     "negative_prompt":negative_prompt
            # })
            # response=requests.request(
            #     "POST",url,headers=headers,data=payload
            # )
            # print(requests)
            # if response.status_code==200:
            #     image_bytes=response.json()["image"].encode()
            #     image_datas=base64.b64decode(image_bytes)




