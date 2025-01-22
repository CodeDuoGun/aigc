import re
from text.symbols2 import punctuation, symbols
from text.tone_sandhi import ToneSandhi
import struct
import uuid
import torchaudio
# from text.g2pw import G2PWPinyin, correct_pronunciation
# from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
import os
import constants
import LangSegment
from text import chinese2, english #import mix_text_normalize,text_normalize,g2p
from log import logger

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open("text/opencpop-strict.txt").readlines()
}
from pypinyin import lazy_pinyin
# g2pw = G2PWPinyin(model_dir="GPT_SoVITS/text/G2PWModel",model_source="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",v_to_u=False, neutral_tone_with_five=True)


tone_modifier = ToneSandhi()
is_g2pw =True


def cut_texts(text:str, how_to_cut:str="按中文句号。切"):
    if (how_to_cut == "凑四句一切"):
        text = cut1(text)
    elif (how_to_cut == "凑50字一切"):
        text = cut2(text)
    elif (how_to_cut == "按中文句号。切"):
        text = cut3(text)
    elif (how_to_cut == "按英文句号.切"):
        text = cut4(text)
    elif (how_to_cut == "按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = text.split("\n")
    texts = process_text(texts) # 有效文本
    texts = merge_short_text_in_array(texts, 5)
    return texts

def clean_special(text, language, special_s, target_symbol, version="v2"):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    if language == "zh":
        norm_text = chinese2.text_normalize(text)
        phones = chinese2.g2p(norm_text)
    else:
        norm_text = english.text_normalize(text)
        phones = english.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    logger.warning(f"new_ph:{new_ph}")
    return new_ph, norm_text


def clean_text(text, language, version="v2"):
    """convert word 2 pinyin"""
    language_module_map = {"zh": "chinese2", "en": "english"}
    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in constants.SPECIAL:
        if special_s in text and language == special_l:
            logger.info(f"cleaning special....")
            return clean_special(text, language, special_s, target_symbol, version)
    if language == "en":
        norm_text = english.text_normalize(text)
        norm_text = text
        logger.info(f"norm text:{text}")
    else:
        norm_text=text
    if language == "zh" or language=="yue":##########
        phones, word2ph = chinese2.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        logger.info(f"norm_text：{norm_text},word2ph:{word2ph}， {len(word2ph)}, {len(norm_text)}")
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = norm_text
        if len(phones) < 4:
            phones = [','] + phones
    else:
        phones = english.g2p(norm_text)
    # phones = ['UNK' if ph not in symbols else ph for ph in phones]
    phones = ['' if ph in ("EE", "AA") else ph for ph in phones]
    return phones, norm_text


def repack_phones(phoneses: list):
    """tool to combine initial and final phone"""
    res = []
    tmp = 0
    tmp_tone = ""
    length=len(phoneses)
    for i in range(length):
        if phoneses[i] in constants.SPLITS:
            res.append(phoneses[i])
            continue
        else:
            tmp_tone += phoneses[i]
            tmp+=1
        if tmp == 2:
            res.append(" ")
            res.append(tmp_tone)
            tmp_tone = ''
            tmp = 0
    return res

def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        Returns:
        List of integers corresponding to the symbols in the text
    '''
    _symbol_to_id_v2 = {s: i for i, s in enumerate(symbols)}
    phones = [_symbol_to_id_v2[symbol] for symbol in cleaned_text]
    return phones

def clean_text_inf(text, language, version):
    # phones = cleaned_text_to_sequence(phones)
    # TODO: @txueduo 这里拼接方式 不对，如果有非字母，应该获取符号后，略过继续拼接下一个
    phones, norm_text = clean_text(text, language, version)
    if language == "zh":
        syllable_phones = repack_phones(phones) #TODO: 英文不应该走这个
    else:
        syllable_phones = phones
    logger.info(f"text:{text},syllable_phones:{syllable_phones}")
    # assert len(syllable_phones) == len(phones)//2
    return syllable_phones, norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in constants.SPLITS:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in constants.SPLITS:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut2(inp):
    """按50字切，未验证"""
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    """按中文句号切分"""
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    """按英文标点符号切，未验证"""
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    """按标点符号切，未验证"""
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []
    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)
    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

def get_phones_and_bert(text,language,version:str="v2",final=False):
    # 把中文符号改成英文
    logger.warning(f"text：{text}")
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            logger.info(f"formattext：{formattext}")
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            logger.info(f"language:{language}, formattext:{formattext}")
            if re.search(r'[A-Za-z%℃]', formattext):
                print(f"中英文混合, :{formattext}")
                formattext = re.sub(r'[a-z]\s*', lambda x: x.group(0), formattext)
                # 循环获取text
                formattext = chinese2.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                logger.info(f"****not A-Za-z， formattext：{formattext}")
                phones, norm_text = clean_text_inf(formattext, language, version)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese2.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, norm_text = clean_text_inf(formattext, language, version)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]

        LangSegment.setfilters(["zh","ja","en","ko"])
        for tmp in LangSegment.getTexts(text):
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
        print(f"****textlist:{textlist}")
        print(f"****langlist:{langlist}")
        phones_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            logger.info(f"textlist[i]:{textlist[i]}")
            phones, norm_text = clean_text_inf(textlist[i], lang, version)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)
    return phones, norm_text


def convert_text2phones(texts: str, text_language:str="en"):
    print(f"实际输入的目标文本(切句后):{texts}")
    res = []
    texts = cut_texts(texts, how_to_cut="按中文句号。切")
    # for text in texts:
    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        text = 
        if (text[-1] not in constants.SPLITS): text += "。" if text_language != "en" else "."
        print(f"实际输入的目标文本(每句):{text}")
        phones2,norm_text2=get_phones_and_bert(text, text_language)
        print(f"前端处理后的文本(每句): {norm_text2}")
        res+=phones2
    # logger.warning(f"res:{res}")
    return res

def convert_decode2bytes(generated_wave,target_sample_rate):
    audio_bytes = generated_wave.numpy().tobytes()
    num_channels = 1  # 假设为单声道
    sample_width = 2  # 假设为 16 位音频，每个样本 2 字节
    sample_rate = target_sample_rate
    audio_bytes = generated_wave.numpy().tobytes()

    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + len(audio_bytes),
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM 格式
        num_channels,
        sample_rate,
        sample_rate * num_channels * sample_width,
        num_channels * sample_width,
        sample_width * 8,
        b'data',
        len(audio_bytes)
    )

    # 组合 wav 头和音频数据
    wav_data = wav_header + audio_bytes
    output_dir = 'tests'
    id = uuid.uuid1()
    saved_audio_path = f"{output_dir}/{id}.wav"
    torchaudio.save(saved_audio_path, generated_wave, target_sample_rate)
    with open(saved_audio_path, 'rb') as f:
        audio_data = f.read()
    return audio_data

if __name__ == "__main__":
    text_language = "all_zh"
    # import nltk
    # nltk.download('averaged_perceptron_tagger_eng')
    # text = "明天有62%的概率降雨，request.remote_addr并不一定能够准确的获得客户端的IP，IP地址，而非用户的真实地址,今天天气多云，32℃，比昨天涨了10%。"
    text = "现在的天气是多云，气温7摄氏度，南风3级，湿度42%。如果你想知道更多天气的信息，可以通过气象台或是气象APP查询。通信距离长度为15km, Automatic speech, 多重任务，多档可调"
    text = "星网一代基本系统，具备通信， 导航增强，探测感知三大类综合信息服务能力，可向苚沪提供移动通信，物联网，宽带通信，航空监视，天基数据回传，通信频段电磁感知和电磁接收等业务，可根据苚沪特殊需求提供定制化服务。星链星座包括一代和二代系统，分别由 一点二万和三万颗味星组成，主要采用K A，KU ，V，E频段。"
    # text = "您好,我是您的客服小助手,能不能方便介绍一下您的身高体重年龄,这样方便为您推荐适合您的款式及尺码."
    # text="Diffusion models proved themselves very effective in artificial synthesis, even beating GANs for images. Because of that, they gained traction in the machine learning community and play an important role for systems like DALL-E 2 or Imagen to generate photorealistic images when prompted on text."
    how_to_cut = "按英文句号.切"
    texts = cut_texts(text, how_to_cut)
    print(f"实际输入的目标文本(切句后):{texts}")
    res = []
    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in constants.SPLITS): text += "。" if text_language != "en" else "."
        print(f"实际输入的目标文本(每句):  {text}")
        phones2,norm_text2=get_phones_and_bert(text, text_language)
        print(f"前端处理后的文本(每句): {norm_text2}")
        res.extend(phones2)
    logger.warning(f"res:{res}")