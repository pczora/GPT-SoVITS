import os
from functools import wraps
from io import BytesIO

import sys
sys.path.append('/root/GPT-SoVITS/GPT_SoVITS')

print("==== sys.path ====")
print(sys.path)

from flask import Flask, make_response, request, jsonify
from werkzeug.utils import secure_filename

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

from api_v2 import pack_audio

UPLOAD_FOLDER = '/workspace/reference_voices'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_prefixed_env('GPTSOVITS_API')

config_path = "GPT_SoVITS/configs/tts_infer.yaml"
tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)



def check_credentials(username, password):
    return username == app.config['BASIC_AUTH_USERNAME'] and password == app.config['BASIC_AUTH_PASSWORD']


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
        return jsonify({'message': 'Bad Request'}), 400
    voice_file = request.files['file']
    if voice_file.filename == '':
        return jsonify({'message': 'Bad Request'}), 400
    if voice_file and allowed_file(voice_file.filename):
        filename = secure_filename(voice_file.filename)
        voice_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'message': 'OK'}), 200


@app.route('/generate', methods=['POST'])
@basic_auth
def generate():
    text = request.json['text']
    voice_name = request.json['voice_name']
    print(voice_name)

    req = {
        "text": text,  # str.(required) text to be synthesized
        "text_lang": "en",  # str.(required) language of the text to be synthesized
        "ref_audio_path": os.path.join(app.config['UPLOAD_FOLDER'], voice_name),  # str.(required) reference audio path
        "aux_ref_audio_paths": [],  # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
        "prompt_text": "",  # str.(optional) prompt text for the reference audio
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
        "streaming_mode": False,  # bool. whether to return a streaming response.
        "parallel_infer": True,  # bool.(optional) whether to use parallel inference.
        "repetition_penalty": 1.35  # float.(optional) repetition penalty for T2S model.
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
