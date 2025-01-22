import torch
import torchaudio
from log import logger
import re
import time
from ema_pytorch import EMA
from model import UNetT, DiT  #,CFM,DiT
from model.cfm_half import CFM
from model.modules import MelSpec
from vocos import Vocos
from config import config
from model.utils import get_tokenizer
from utils.tool import convert_text2phones, cut_texts
from einops import rearrange

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "Emilia_ZH_EN"

# TODO: 移动到常量
nfe_step = 16  # 16, 32
cfg_strength = 2.
ode_method = 'euler'  # euler | midpoint
sway_sampling_coef = -1.
speed = 1.
fix_duration = None  # 27  # None (will linear estimate. if code-switched, consider fix) | float (total in seconds, include ref audio) 


n_mel_channels = 100
hop_length = 256
target_rms = 0.1

# mel_spec = default(None, MelSpec(**mel_spec_kwargs))
def load_vocos(local:bool=True):
    # 音频处理和文本处理
    if local:
        vocos_local_path = "./ckpts/charactr/vocos-mel-24khz"
        vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
        vocos.load_state_dict(state_dict)
        vocos.eval()
    else:
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocos



def load_model(use_ema:bool=True,exp_name:str = "F5TTS_Base",ckpt_step:int=1200000,tokenizer:str="pinyin",target_sample_rate:int=24000):
    """"""
    checkpoint = torch.load(f"ckpts/{exp_name}/model_{ckpt_step}.pt", map_location=device)
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
    mel_spec_kwargs=dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    )
    # 
    if exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

    elif exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
        )
    model = model.half().to(device)
    if use_ema == True:
        ema_model = EMA(model, include_online_model=False).to(device)
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        ema_model.copy_params_from_ema_to_model()
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def gen_audio(model, vocos, ref_map: dict, ref_id: str, ref_text:str, gen_text: str, tokenizer:str="pinyin"):
    """Tool to gen audio by ref_audio and ref_text
    Args:
        ref_map
    """
    ref_audio = ref_map[ref_id]['ref_audio']
    sr = ref_map[ref_id]['ref_sr']
    rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
    if rms < target_rms:
        ref_audio = ref_audio * target_rms / rms
    if sr!= config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        ref_audio = resampler(ref_audio)

    texts = ref_text + gen_text
    # 切割分段
    # texts = cut_texts(texts, how_to_cut="按中文句号。切")
    # for text in texts:
    if tokenizer == "pinyin":
        # final_text_list = convert_char_to_pinyin_avcc(text_list)
        final_text_list = [convert_text2phones(texts,  text_language="all_zh")]
    else:
        final_text_list = [texts]
    logger.info(f"******final_text_list:{final_text_list}")
    # calculate duration #TODO: @txueduo， 优化这部分
    ref_audio_len = ref_audio.shape[-1] // hop_length
    if fix_duration is not None:
        duration = int(fix_duration * config.SAMPLE_RATE / hop_length)
    else:  # simple linear scale calcul
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
        # ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        # gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

    # 生成音频
    starttime = time.time()
    with torch.inference_mode():
        ref_audio = ref_audio.to('cuda')
        generated, trajectory = model.sample(
            cond=ref_audio.half(),
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=None
        )

    generated = generated[:, ref_audio_len:, :]
    generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms
    logger.info(f"generated_wave type:{type(generated_wave)}")
    return generated_wave
