import os
import base64
import re
from datetime import date

import sounddevice as sd
import numpy as np
import pyttsx3

from faster_whisper import WhisperModel
from openai import OpenAI
from dotenv import load_dotenv

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

FS = 16000
DURATION = 5

print("Chargement de Whisper...")
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

client = OpenAI(
    api_key=os.environ.get("GROQ_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def parler(texte):
    tts = pyttsx3.init("sapi5")

    voices = tts.getProperty("voices")

    if voices:
        tts.setProperty("voice", voices[0].id)

    tts.setProperty("rate", 150)
    tts.setProperty("volume", 1.0)

    print("🔊 Lecture vocale...")

    tts.say(texte)
    tts.runAndWait()

    print("🔊 Lecture terminée.")

    tts.stop()

def connecter_gmail():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


def extraire_texte(part):

    resultat = ""

    if part.get("mimeType") == "text/plain":

        body = part.get("body", {})

        if "data" in body:

            try:
                resultat += base64.urlsafe_b64decode(
                    body["data"]
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

            except Exception:
                pass

    for sous_partie in part.get("parts", []):
        resultat += extraire_texte(sous_partie)

    return resultat


def recuperer_mails(service):

    aujourd_hui = date.today().strftime("%Y/%m/%d")

    result = service.users().messages().list(
        userId="me",
        q=f"after:{aujourd_hui}"
    ).execute()

    messages = result.get("messages", [])

    mails = []

    for message in messages:

        mail = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        payload = mail.get("payload", {})

        headers = payload.get("headers", [])

        expediteur = "Inconnu"
        sujet = "Sans sujet"

        for header in headers:

            if header["name"].lower() == "from":
                expediteur = header["value"]

            elif header["name"].lower() == "subject":
                sujet = header["value"]

        contenu = extraire_texte(payload)

        if not contenu:

            body = payload.get("body", {})

            if "data" in body:

                try:
                    contenu = base64.urlsafe_b64decode(
                        body["data"]
                    ).decode(
                        "utf-8",
                        errors="ignore"
                    )

                except Exception:
                    contenu = ""

        mails.append({
            "expediteur": expediteur,
            "sujet": sujet,
            "contenu": contenu
        })

    return mails

def normaliser_demande(texte):

    demande = texte.lower()

    demande = demande.replace("mailles", "mails")
    demande = demande.replace("émails", "emails")
    demande = demande.replace("e-mails", "emails")
    demande = demande.replace("e mail", "email")
    demande = demande.replace("e mails", "emails")

    return demande

def trouver_mail(demande, mails):

    ordres = {
        "premier": 0,
        "première": 0,
        "deuxième": 1,
        "second": 1,
        "seconde": 1,
        "troisième": 2,
        "quatrième": 3,
        "cinquième": 4
    }

    for mot, index in ordres.items():

        if mot in demande:

            if index < len(mails):
                return mails[index]

            return None

    mots = re.findall(
        r"[a-zA-ZÀ-ÿ0-9@._-]+",
        demande
    )

    mots_inutiles = {
        "lis",
        "lire",
        "le",
        "la",
        "les",
        "mail",
        "mails",
        "email",
        "emails",
        "de",
        "du",
        "des",
        "qui",
        "parle",
        "résume",
        "résumer",
        "moi",
        "ce",
        "cet",
        "cette",
        "sur",
        "dans",
        "un",
        "une",
        "mes",
        "mon",
        "ma"
    }

    mots = [
        mot for mot in mots
        if mot not in mots_inutiles
        and len(mot) > 2
    ]

    resultats = []

    for mail in mails:

        texte_mail = (
            mail["expediteur"]
            + " "
            + mail["sujet"]
            + " "
            + mail["contenu"]
        ).lower()

        score = 0

        for mot in mots:

            if mot in texte_mail:
                score += 1

        if score > 0:
            resultats.append(
                (score, mail)
            )

    if not resultats:
        return None

    resultats.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if len(resultats) > 1:

        if resultats[0][0] == resultats[1][0]:
            return None

    return resultats[0][1]

def resumer_mail(mail):

    contenu = mail["contenu"]

    if len(contenu) > 12000:
        contenu = contenu[:12000]

    prompt = f"""
Tu es un assistant vocal qui aide à gérer les emails.

Voici un email :

Expéditeur :
{mail["expediteur"]}

Sujet :
{mail["sujet"]}

Contenu :
{contenu}

Résume cet email en français.

Règles :
- réponse courte
- naturelle à l'oral
- pas de Markdown
- ignore les signatures
- ignore les liens inutiles
- explique clairement ce que veut l'expéditeur
- indique s'il y a une action importante à effectuer
"""

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    return response.output_text.strip()

def resumer_tous_les_mails(mails):

    resultats = []

    for i, mail in enumerate(mails, start=1):

        resume = resumer_mail(mail)

        resultats.append(
            f"Mail {i} : {resume}"
        )

    return "\n\n".join(resultats)

def liste_mails(mails):

    if not mails:
        return "Tu n'as reçu aucun mail aujourd'hui."

    texte = f"Tu as reçu {len(mails)} mails aujourd'hui."

    for i, mail in enumerate(mails, start=1):

        texte += (
            f" Mail {i}, envoyé par "
            f"{mail['expediteur']}, "
            f"avec pour sujet "
            f"{mail['sujet']}."
        )

    return texte

def traiter_demande(texte, mails):

    demande = normaliser_demande(texte)

    if (
        "combien" in demande
        or "nombre" in demande
    ):

        if len(mails) == 0:
            return "Tu n'as reçu aucun mail aujourd'hui."

        return (
            f"Tu as reçu {len(mails)} "
            f"mails aujourd'hui."
        )

    if (
        "quels" in demande
        or "liste" in demande
        or "montre" in demande
        or "voir" in demande
        or "consulte" in demande
        or "regarde" in demande
    ):

        return liste_mails(mails)

    if (
        "résume mes mails" in demande
        or "résumer mes mails" in demande
        or "résumé de mes mails" in demande
        or "lis mes mails" in demande
        or "lire mes mails" in demande
        or "important" in demande
        or "parle-moi de mes mails" in demande
    ):

        if not mails:
            return "Tu n'as reçu aucun mail aujourd'hui."

        return resumer_tous_les_mails(mails)

    mots_selection = [
        "ce mail",
        "cet email",
        "le mail de",
        "le mail qui",
        "lis le mail",
        "lire le mail",
        "résume le mail",
        "résumer le mail",
        "premier mail",
        "deuxième mail",
        "troisième mail",
        "quatrième mail",
        "cinquième mail"
    ]

    selection = any(
        mot in demande
        for mot in mots_selection
    )

    if selection:

        mail = trouver_mail(
            demande,
            mails
        )

        if mail is None:

            return (
                "Je n'ai pas réussi à déterminer "
                "quel mail tu veux lire."
            )

        return resumer_mail(mail)

    prompt = f"""
Tu es un assistant vocal personnel.

Tu reçois une demande de l'utilisateur sous forme
de texte transcrit depuis sa voix.

Réponds en français,
de manière naturelle,
claire et concise.

Si la demande concerne les emails mais que tu ne
peux pas déterminer lequel, demande une précision.

N'invente jamais d'informations.

Demande de l'utilisateur :
{texte}
"""

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    return response.output_text.strip()

def ecouter():

    print("Parle maintenant...")

    data = sd.rec(
        int(DURATION * FS),
        samplerate=FS,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    mono = data[:, 0]

    audio = mono.astype(
        "float32"
    ) / 32768.0

    if audio.max() - audio.min() < 0.02:
        return ""

    segments, info = model.transcribe(
        audio,
        language="fr",
        vad_filter=True
    )

    texte = ""

    for segment in segments:
        texte += segment.text

    return texte.strip()

print("Connexion à Gmail...")

service = connecter_gmail()

mails = recuperer_mails(service)

print(
    f"Tu as reçu {len(mails)} "
    f"mails aujourd'hui."
)

parler(
    f"Salut, tu as reçu {len(mails)} "
    f"mails aujourd'hui."
)

continuer = True

while continuer:

    texte = ecouter()

    if not texte:
        continue

    print(texte)

    if texte.lower().strip() == "exit":
        parler("À bientôt.")
        continuer = False
        break

    reponse = traiter_demande(
        texte,
        mails
    )

    print(reponse)

    parler(reponse)
