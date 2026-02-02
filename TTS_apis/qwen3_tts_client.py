import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
#
# 1. Qwen3-TTS는 sox가 필요하다.
#    다음 링크에서 sox 14.4.2를 다운로드 받는다.
#    https://sourceforge.net/projects/sox/files/sox/
#    PATH에 sox 경로를 추가한다. (기본값은 C:\Program Files (x86)\sox-14-4-2)
#
# 2. Qwen3-TTS 패키지 설치
#    AlwaysReddy\venv\Scripts\activate.bat를 실행하여 가상 환경 안에서 qwen-tts를 설치한다.
# > pip install -U qwen-tts
#
# 3. flash-attn을 사용하면 Nvidia 3000번대 이후 GPU에서 VRAM 사용량을 아낄 수 있지만, 설치 중 컴파일 에러가 발생해서 당분간 설치 보류.
#    시도해 보려면 다음을 참고한다.
#    Visual Studio Build Tools를 설치한 후, Developer Command Prompt를 띄워서 컴파일 환경을 설정한 후,
#    AlwaysReddy 가상 환경(activate.bat 실행)에서 설치 작업을 진행한다.
# > set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<버전>"
# > set DISTUTILS_USE_SDK=1
# > pip install -U flash-attn --no-build-isolation
# (CUDA_HOME 환경변수를 설정했는 데도 CUDA_HOME 에러가 발생하면 다음 링크를 참조하여 PyTorch를 CPU-only버전이 아닌 CUDA버전으로 재설치.
#  https://pytorch.org/get-started/locally/)
#
class Qwen3TTSClient:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )

    def __init__(self, verbose=False):
        """Initialize the Qwen3 TTS client."""
        self.verbose = verbose

    def tts(self, text_to_speak, output_file):
        """
        This function uses the Qwen3 TTS engine to convert text to speech.

        Args:
            text_to_speak (str): The text to be converted to speech.
            output_file (str): The path where the output audio file will be saved.

        Returns:
            str: "success" if the TTS process was successful, "failed" otherwise.
        """

        try:
            wavs, sr = Qwen3TTSClient.model.generate_custom_voice(
                text=text_to_speak,
                language="Korean", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
                speaker="Sohee",
                # instruct="상냥하고 부드럽게", # Omit if not needed.
            )

            sf.write(output_file, wavs[0], sr)

            return "success"
        except Exception as e:
            # If the command fails, print an error message and return "failed"
            if self.verbose:
                print(f"Error running Qwen3 TTS command: {e}")
            return "failed"