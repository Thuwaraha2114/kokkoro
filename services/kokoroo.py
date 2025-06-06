from kokoro import KPipeline
import torch

class GetAudio:
    def __init__(self, text):
        self.text = text
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.pipeline = KPipeline(lang_code='b', device=self.device)

    def generate_audio(self):
        generator = self.pipeline(
            self.text,
            voice='hf_beta,af_nicole',
            speed=1.1,
            split_pattern=r'[.!?]'
        )

        for _, _, audio in generator:
            yield audio
