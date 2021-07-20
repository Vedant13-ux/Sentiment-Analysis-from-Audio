import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import torch

class Model:
    def __init__(self, filename):
        self.recognizer=sr.Recognizer()
        self.filename=filename

    def change_format(self):
        if(self.filename.split('.')[1]!='wav'):
            print('Already a WAV File')
            self.filename=f"{self.filename.split('.')[0]}.wav"
            return 

        audio = AudioSegment.from_file(f'./audio/{self.filename}', format=self.filename.split('.')[1])
        self.filename=f"{self.filename.split('.')[0]}.wav"
        audio.export(f'./audio/{self.filename}', format="wav")

    def audio_to_text(self):
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        speech, rate = librosa.load(f'./audio/{self.filename}', sr=16000)

        input_values = tokenizer(speech, return_tensors='pt').input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        self.transcript=transcription
        print(transcription)
        return transcription
    
    def sentiment_analysis(self):
        classifier = pipeline('sentiment-analysis')
        results = classifier(self.transcript)[0]
        label=results['label']
        score=results['score']
        print(label, score)
        return results  
    
    def return_filename(self):
        return self.filename
if __name__=="__main__":
    model= Model('sound.wav')
    model.change_format()
    model.audio_to_text()
    model.sentiment_analysis()
