import os
import subprocess
from functools import wraps
from io import BytesIO

import flask.wrappers
import numpy as np
import soundfile as sf
from flask import Flask, make_response, request, jsonify
from werkzeug.utils import secure_filename

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

UPLOAD_FOLDER = '/workspace/reference_voices'
ALLOWED_EXTENSIONS = {'wav', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_prefixed_env('GPTSOVITS_API')





def init_config() -> TTS_Config:
    # config_path = "GPT_SoVITS/configs/tts_infer.yaml"
    # conf = TTS_Config(config_path)
    conf = TTS_Config()
    if "BERT_BASE_PATH" in app.config:
        conf.bert_base_path = app.config["BERT_BASE_PATH"]
    if "CNHUBERT_BASE_PATH" in app.config:
        conf.cnhuhbert_base_path = app.config["CNHUBERT_BASE_PATH"]
    if "DEVICE" in app.config:
        conf.device = app.config["DEVICE"]
    if "T2S_WEIGHT_PATH" in app.config:
        conf.t2s_weights_path = app.config["T2S_WEIGHT_PATH"]
    if "VERSION" in app.config:
        conf.version = app.config["VERSION"]

    print(conf)
    return conf

tts_config = init_config()
tts_pipeline = TTS(tts_config)

defaults = {
    "text_lang": "en",
    "prompt_lang": "en",  # str.(required) language of the prompt text for the reference audio
    "top_k": 5,  # int. top k sampling
    "top_p": 1,  # float. top p sampling
    "temperature": 1,  # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,  # int. batch size for inference
    "batch_threshold": 0.75,  # float. threshold for batch splitting.
    "split_bucket": True,  # bool. whether to split the batch into multiple buckets.
    "speed_factor": 1.0,  # float. control the speed of the synthesized audio.
    "fragment_interval": 0.3,  # float. to control the interval of the audio fragment.
    "seed": -1,  # int. random seed for reproducibility.
    "media_type": "wav",  # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
    "parallel_infer": True,  # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35  # float.(optional) repetition penalty for T2S model.
}


def check_credentials(username, password):
    username_check = not app.config['BASIC_AUTH_USERNAME'] or username == app.config['BASIC_AUTH_USERNAME']
    password_check = not app.config['BASIC_AUTH_PASSWORD'] or password == app.config['BASIC_AUTH_PASSWORD']
    return  username_check and password_check

def basic_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_credentials(auth.username, auth.password):
            return jsonify({'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)

    return decorated_function


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_voice', methods=['POST'])
@basic_auth
def upload_voice():
    if 'file' not in request.files:
        return jsonify({'message': 'Bad Request; file not present'}), 400
    voice_file = request.files['file']
    if voice_file.filename == '':
        return jsonify({'message': 'Bad Request; file name not present'}), 400
    if voice_file and allowed_file(voice_file.filename):
        filename = secure_filename(voice_file.filename)
        voice_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if 'reference_text' in request.files:
            reference_text_file = request.files['reference_text']
            reference_text_filename = voice_file.filename.replace(".wav", ".txt")
            reference_text_file.save(os.path.join(app.config['UPLOAD_FOLDER'], reference_text_filename))
    return jsonify({'message': 'OK'}), 200


@app.route('/generate', methods=['POST'])
@basic_auth
def generate():
    text = request.json['text']
    voice_name = str(request.json['voice_name'])
    print(voice_name)

    reference_text_path = os.path.join(app.config['UPLOAD_FOLDER'], voice_name.replace(".wav", ".txt"))
    reference_text = ""
    if os.path.isfile(reference_text_path):
        f = open(reference_text_path, "r")
        reference_text = f.read()
        f.close()


    text_lang = from_request_or_default('text_lang', request)
    prompt_lang = from_request_or_default('prompt_lang', request)
    top_k = from_request_or_default('top_k', request)
    top_p = from_request_or_default('top_p', request)
    temperature = from_request_or_default('temperature', request)
    text_split_method = from_request_or_default('text_split_method', request)
    batch_size = from_request_or_default('batch_size', request)
    batch_threshold = from_request_or_default('batch_threshold', request)
    split_bucket = from_request_or_default('split_bucket', request)
    speed_factor = from_request_or_default('speed_factor', request)
    fragment_interval = from_request_or_default('fragment_interval', request)
    seed = from_request_or_default('seed', request)
    media_type = from_request_or_default('media_type', request)
    parallel_infer = from_request_or_default('parallel_infer', request)
    repetition_penalty = from_request_or_default('repetition_penalty', request)


    req = {
        "text": text,  # str.(required) text to be synthesized
        "text_lang": text_lang,  # str.(required) language of the text to be synthesized
        "ref_audio_path": os.path.join(app.config['UPLOAD_FOLDER'], voice_name),  # str.(required) reference audio path
        "aux_ref_audio_paths": [],  # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
        "prompt_text": reference_text,  # str.(optional) prompt text for the reference audio
        "prompt_lang": prompt_lang,  # str.(required) language of the prompt text for the reference audio
        "top_k": top_k,  # int. top k sampling
        "top_p": top_p,  # float. top p sampling
        "temperature": temperature,  # float. temperature for sampling
        "text_split_method": text_split_method,  # str. text split method, see text_segmentation_method.py for details.
        "batch_size": batch_size,  # int. batch size for inference
        "batch_threshold": batch_threshold,  # float. threshold for batch splitting.
        "split_bucket": split_bucket,  # bool. whether to split the batch into multiple buckets.
        "speed_factor": speed_factor,  # float. control the speed of the synthesized audio.
        "fragment_interval": fragment_interval,  # float. to control the interval of the audio fragment.
        "seed": seed,  # int. random seed for reproducibility.
        "media_type": media_type,  # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
        "streaming_mode": False,  # bool. whether to return a streaming response.
        "parallel_infer": parallel_infer,  # bool.(optional) whether to use parallel inference.
        "repetition_penalty": repetition_penalty  # float.(optional) repetition penalty for T2S model.
    }
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        sr, audio_data = next(tts_generator)
        audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        response = make_response(audio_data)
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
        return response
    except Exception as e:
        return jsonify({"message": f"tts failed", "Exception": str(e)}), 400


def from_request_or_default(key: str, req: flask.wrappers.Request):
    if key in req.json:
        return req.json[key]
    elif key in defaults:
        return defaults[key]
    else:
        return None

def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5555,  debug=True)