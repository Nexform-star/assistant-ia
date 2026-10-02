import sounddevice as sd
import wave



print(dir(wave))
fs = 44100
duration = 5

data = sd.rec(int(duration * fs), channels=2, dtype="int16")
sd.wait()
with wave.open("audio.wav", "wb") as wav_file:
    wav_file.setnchannels(2)
    wav_file.setsampwidth(2)
    wav_file.setframerate(44100)
    wav_file.writeframes(data.tobytes())
print(data.shape)
print(data.dtype)

