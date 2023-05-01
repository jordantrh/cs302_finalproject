/*

Make sure your python version is > 3.11. 
This can be found by typing "python --version" in commandline

*/

import openai
import whisper

model = whisper.load_model("medium")
result = model.transcribe("./test.mp3")
print(result["text"])
