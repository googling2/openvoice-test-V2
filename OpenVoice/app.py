
import gradio as gr
import os, torch, io

import se_extractor
from api import ToneColorConverter

from melo.api import TTS

# from resemble_enhance.enhancer.inference import denoise, enhance

import torchaudio
import soundfile as sf
import gc



reference_wavs = ["참조 오디오를 선택하거나 직접 업로드하세요."]

for name in os.listdir("./audio/"):
    reference_wavs.append(name)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def change_choices():

    reference_wavs = ["참조 오디오를 선택하거나 직접 업로드하세요."]

    for name in os.listdir("./audio/"):
        reference_wavs.append(name)
    
    return {"choices":reference_wavs, "__type__": "update"}


def change_wav(audio_path):

    return f"./audio/{audio_path}"

def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def _fn(path, solver="Midpoint", nfe=64, tau=0.5,chunk_seconds=10,chunks_overlap=0.5, denoising=True):
    
    if path is None:
        return None, None
    print(path)
    sf.write('./output.wav', path[1], path[0], 'PCM_24')

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load('./output.wav')
    dwav = dwav.mean(dim=0)

    wav2, new_sr = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

    wav2 = wav2.cpu().numpy()

    clear_gpu_cash()
    return (new_sr, wav2)

def reference(text,speed,wav,language):

    ckpt_converter = './checkpoints_v2/converter'
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = './outputs_v2'

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    reference_speaker = wav # This is the voice you want to clone
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)


    src_path = f'{output_dir}/tmp.wav'

    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'./checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)

    return save_path

    


def main():
    with gr.Blocks() as demo:
        gr.Markdown('# OpenVoiceV2 WebUI\n\nA WebUI for OpenVoiceV2.')
        with gr.Group():
            language = gr.Radio(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], label='언어', value='KR')
            speed = gr.Slider(label="말하는 속도 조정", minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
            wavs_dropdown = gr.Dropdown(label="오디오 목록",choices=reference_wavs,value="참조 오디오를 선택하거나 직접 업로드하세요.",interactive=True)
            refresh_button = gr.Button("오디오목록 새로고침")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[wavs_dropdown])
            a_aud = gr.Audio(label="참조 오디오", type="filepath")
            wavs_dropdown.change(change_wav,[wavs_dropdown],[a_aud])
            text = gr.Textbox(label="추론텍스트",value="", lines=16, max_lines=16)

        
        btn = gr.Button('생성', variant='primary')
        aud = gr.Audio(label="결과물",show_download_button=True)
        up_button = gr.Button("오디오 소음 감소 향상")
        up_button.click(_fn, [aud], [aud])
        btn.click(reference, inputs=[text,speed,a_aud,language], outputs=[aud])
        gr.Markdown('googling team iforum project.')


    demo.queue().launch(inbrowser=True,server_name="0.0.0.0",)

if __name__ == "__main__":
    main()







