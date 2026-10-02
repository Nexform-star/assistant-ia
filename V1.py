import sounddevice as sd
import wave
from faster_whisper import WhisperModel
from openai import OpenAI
import os
from dotenv import load_dotenv



load_dotenv()
fs = 16000
duration = 5
prompt = """
Tu es un assistant vocal personnel.

Tu reçois les messages de l'utilisateur sous forme de texte transcrit depuis sa voix.
Réponds en français, de manière naturelle, claire et concise.
Si la demande est ambiguë, demande une précision.
N'invente jamais d'informations.

"""

model = WhisperModel("small", device="cpu", compute_type="int8")

client = OpenAI(
    api_key=os.environ.get("GROQ_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

continuer = True

while continuer:

    print("Parle maintenant...")

    data = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    with wave.open("audio.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs)
        wav_file.writeframes(data.tobytes())

    print(data.shape)
    print(data.dtype)

    mono = data[:, 0]
    audio = mono.astype("float32") / 32768.0

    print(audio.min())
    print(audio.max())

    segments, info = model.transcribe(
        audio,
        language="fr"
    )

    print(info)

    for segment in segments:

        texte = segment.text.strip()

        print(texte)

        if texte.lower() == "exit":
            continuer = False
            break
        message = prompt + "\n" + texte
        response = client.responses.create(
            input=message,
            model="openai/gpt-oss-20b",
        )

        print(response.output_text)

print("Assistant arrêté.")