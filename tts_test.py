import pyttsx3

engine = pyttsx3.init()  # try default driver
engine.setProperty('rate', 170)

# Set volume explicitly
engine.setProperty('volume', 1.0)

# Optional: choose a voice and print info
voices = engine.getProperty('voices')
print("Available voices:")
for i, v in enumerate(voices):
    print(i, v.id)

# Pick a different voice to test (0 or 1 usually)
if voices:
    engine.setProperty('voice', voices[0].id)

print("Speaking test...")
engine.say("Hello Mr. Ashish!")
engine.runAndWait()
print("Done.")