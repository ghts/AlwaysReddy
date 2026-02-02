from supertonic import TTS

# https://supertone-inc.github.io/supertonic-py/
#
# 1.Supertonic2-TTS 패키지 설치
#    AlwaysReddy\venv\Scripts\activate.bat를 실행하여 가상 환경 안에서 supertonic2-tts를 설치한다.
# > pip install supertonic
#
class Supertonic2TTSClient:
    model = None
    style = None

    def __init__(self, verbose=False):
        """Initialize the Supertonic2 TTS client."""
        self.verbose = verbose

        if Supertonic2TTSClient.model is None:
            # Note: First run downloads model automatically (~305MB)
            Supertonic2TTSClient.model = TTS(auto_download=True)

            # Get a voice style
            Supertonic2TTSClient.style = Supertonic2TTSClient.model.get_voice_style(voice_name="F1")

    def tts(self, text_to_speak, output_file):
        """
        This function uses the Supertonic2 TTS engine to convert text to speech.

        Args:
            text_to_speak (str): The text to be converted to speech.
            output_file (str): The path where the output audio file will be saved.

        Returns:
            str: "success" if the TTS process was successful, "failed" otherwise.
        """
        try:
            wav, _ = Supertonic2TTSClient.model.synthesize(text_to_speak, voice_style=Supertonic2TTSClient.style, lang="ko")
            Supertonic2TTSClient.model.save_audio(wav, output_file)

            return "success"
        except Exception as e:
            # If the command fails, print an error message and return "failed"
            if self.verbose:
                print(f"Error running Supertonic2 TTS command: {e}")
            return "failed"





