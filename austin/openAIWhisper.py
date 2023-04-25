import openai
import whisper

model = whisper.load_model("base.en")
result = model.transcribe("./test.mp3")
print(result["text"])