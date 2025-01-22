from log import logger
from config import config

from flask import Flask, Response, request
import os
import time
import json
import torch
import torchaudio

from model.modules import MelSpec
from utils.tool import convert_decode2bytes
from service.audio import load_model, load_vocos, gen_audio
app = Flask(__name__)
# 全局字典用于存储数据


ref_map = {}
model = load_model()
vocos = load_vocos()


@app.route('/generate_audio', methods=['GET'])
def generate_audio():
    starttime = time.time()
    data = request.get_json()
    ref_id = data.get('ref_id')
    gen_text = data.get('gen_text')
    logger.info(f"will generate audio {ref_id} {gen_text}")
    # logger.info(ref_id,ref_map.keys())
    if ref_id not in ref_map:
        return "Invalid ref_id", 400
    
    # ref_audio_path = ref_map[ref_id]['ref_audio_path']
    # ref_text = ref_map[ref_id]['ref_text']
    # 您的原始配置和参数设置

    # ref_audio = ref_audio_path #"tests/ref_audio/azure_fc82eedc-86d8-11ef-8b50-61e25f94ca61.wav"
    #ref_text = "您好，我是您的客服小助手，能不能方便介绍一下您的身高体重年龄，这样方便为您推荐适合您的款式及尺码。"
    # gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

    # 初始化模型
    # audio, sr = torchaudio.load(ref_audio)
    # logger.info(ref_map[ref_id])
    generated_wave = gen_audio(model, vocos, ref_map, ref_id, ref_map[ref_id]['ref_text'], gen_text)

    # 直接返回音频数据
    audio_data = convert_decode2bytes(generated_wave, config.SAMPLE_RATE)
    logger.info(f" Generated audio for {ref_id} {gen_text}")
    return Response(audio_data, mimetype='audio/wav')
import hashlib

def get_md5(s):
    m = hashlib.md5()
    m.update(s.encode('utf-8'))
    return m.hexdigest()

def get_file_md5(file_path):
    if not os.path.isfile(file_path):
        return "File does not exist"
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
FIELD_NAME = 'ref_audio'
@app.route('/submit_data', methods=['POST'])
def submit_data():
    ref_audio = request.files['ref_audio']
    ref_text = request.form['ref_text']

    ref_id = get_md5(ref_text)

    # 保存音频到本地
    audio_path = os.path.join('audio_files', f'{ref_id}.wav')
    ref_audio.save(audio_path)
    audio_md5 = get_file_md5(audio_path)
    audio_path_new = os.path.join('audio_files', f'{audio_md5}.wav')
    os.rename(audio_path,audio_path_new)
    ref_map[audio_md5] = {
        'ref_audio_path': audio_path_new,
        'ref_text': ref_text
    }
    load_ref_audio(audio_md5,audio_path_new)
    # 保存数据字典到本地 JSON 文件
    # logger.info(ref_map)
    to_be_save_dict = {}
    with open('data.json', 'w') as f:
        for key in ref_map.keys():
            new_dict = {k: (v.tolist() if k == FIELD_NAME and isinstance(v, torch.Tensor) else v) for k, v in ref_map[key].items()}
            to_be_save_dict[key] = new_dict
        json.dump(to_be_save_dict, f)

    return audio_md5

def load_ref_audio(key,audio_path):
    audio, sr = torchaudio.load(audio_path)
    ref_map[key]["ref_audio"] = audio
    ref_map[key]["ref_sr"] = sr

if __name__ == '__main__':
    if not os.path.exists('audio_files'):
        os.makedirs('audio_files')
    # app.run(debug=True)
    if os.path.exists('data.json'):
        with open('data.json', 'r') as f:
            ref_map = json.load(f)
            for key in ref_map.keys():
                new_dict = {k: (torch.tensor(v) if k == FIELD_NAME and isinstance(v, list) else v) for k, v in ref_map[key].items()}
                ref_map[key] = new_dict
    else:
        ref_map = {}
    # for key in ref_map.keys():
    #     load_ref_audio(ref_map[key]["ref_audio_path"])
    logger.info(f" all voice tone has been loaded {ref_map.keys()}")
    app.run(host=config.HOST, port=config.PORT)