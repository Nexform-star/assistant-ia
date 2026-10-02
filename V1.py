import sounddevice as sd
import wave
from faster_whisper import WhisperModel

fs = 16000
duration = 5

model = WhisperModel("small", device="cpu", compute_type="int8")

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

print("Assistant arrêté.")