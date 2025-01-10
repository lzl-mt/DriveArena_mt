import base64
import concurrent.futures
import copy
import pickle
import time
from functools import partial

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/jfs/dong.yang/cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("/jfs/dong.yang/cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)


def get_openai_description(image_path):
    # Getting the base64 string
    messages = [
	    {
		    "role": "user",
		    "content": [
			    {
				    "type": "image",
				    "image": f"file://{image_path}",
			    },
			    {"type": "text", "text": f"""
                Based on the driving image, you need to give the following CORE information about it:
                - Time of the day: daytime or night.
                - Weather: Sunny, rainy, cloudy, or snowy
                - Surrounding environment: downtown, suburban, rural, or nature.
                - Road condition: (The car is driving on) intersection/straight road/narrow street/wide road/ped crossing/ etc.
                - Give 2-3 key words to desctibe other key infomation about surrounding, especially colors of the buildings/vehicles/trees.
                Your answer should be several keywords seperate by commas, no need for a sentence. e.g "daytime, cloudy, nature, wide street, white building, green trees."
                """},
		    ],
	    }
    ]

    text = processor.apply_chat_template(
	    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
	    text=[text],
	    images=image_inputs,
	    videos=video_inputs,
	    padding=True,
	    return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [
	    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
	    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def process_scene(scene, token_data_dict, data_infos, get_openai_description):
    idx = int(len(scene) / 2)
    image_path = data_infos[token_data_dict[scene[idx]]]["cams"]["CAM_FRONT"]["data_path"]
    description = get_openai_description(image_path)
    print(description)
    return scene, description


file = open(
    "data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl",
    "rb",
)
datas = pickle.load(file)
data_infos = list(sorted(datas["infos"], key=lambda e: e["timestamp"]))
data_infos = data_infos[::1]
scene_tokens = datas["scene_tokens"]
token_data_dict = {item["token"]: idx for idx, item in enumerate(data_infos)}
print(f'Available scenes: {len(scene_tokens)}')

with concurrent.futures.ThreadPoolExecutor(
    max_workers=1
) as executor:
    process_scene_partial = partial(
        process_scene,
        token_data_dict=token_data_dict,
        data_infos=data_infos,
        get_openai_description=get_openai_description,
    )
    results = list(executor.map(process_scene_partial, scene_tokens))
for i, (scene, description) in enumerate(results):
    print(f"scene {i}", description)
    for token in scene:
        data_infos[token_data_dict[token]]["description"] = description

with open(
    "data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val_with_note.pkl",
    "wb",
) as f:
    data_copy = copy.deepcopy(datas)
    data_copy["infos"] = data_infos
    pickle.dump(data_copy, f)
