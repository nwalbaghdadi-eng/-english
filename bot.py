#!/usr/bin/env python3
"""
Englizy v2 - an English tutor Telegram bot for Egyptian Arabic speakers.

What's new in v2:
  - Send a VOICE NOTE: the bot listens, transcribes, corrects your
    pronunciation gently (in Egyptian Arabic) and replies with its own
    voice note so you hear the correct pronunciation.
  - Send a PHOTO or a short VIDEO: the bot describes it in simple English
    and talks with you about it. (Live camera is not possible on Telegram.)
  - Beginner pacing: the bot starts with child-simple English and levels
    up gradually. Change it any time with /level.

Commands:
  /start   welcome message + resets the conversation
  /help    what the bot can do
  /quiz    a quick vocabulary quiz (multiple choice)
  /cancel  stop the current quiz
  /level   show or set your level: beginner / easy / medium

Configuration (environment variables, or a .env file next to this file):
  TELEGRAM_BOT_TOKEN  (required) token from @BotFather
  LLM_PROVIDER        "gemini" (default) or "openai" (text chat only)
  GEMINI_API_KEY      required when LLM_PROVIDER=gemini
  GEMINI_MODEL        default: gemini-3.8-flash (free tier, multimodal)
  OPENAI_API_KEY      required when LLM_PROVIDER=openai
  OPENAI_BASE_URL     optional; set it for OpenAI-compatible providers
  OPENAI_MODEL        default: gpt-4o-mini
  TTS_VOICE           default: en-US-JennyNeural (free, no API key needed)
  TTS_RATE            default: -20% (slower = easier to follow)
  PORT                default: 8080 - tiny health page for free hosting
"""

import logging
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import io

import edge_tts
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("englizy")

TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "-20%")
MAX_HISTORY_MESSAGES = 30  # remembered back-and-forth messages per user

# ---------------------------------------------------------------------------
# Levels: the bot starts at "beginner" and can be changed with /level.
# ---------------------------------------------------------------------------
LEVELS = ("beginner", "easy", "medium")

LEVEL_STYLE = {
    "beginner": (
        "The learner is an ABSOLUTE beginner, like a child learning to talk. "
        "Use very short sentences (3-6 words), only everyday words, simple "
        "present tense. After any new word, add its meaning in Egyptian "
        "Arabic in parentheses. Be very patient and encouraging."
    ),
    "easy": (
        "The learner knows the basics. Use short sentences, common words, "
        "simple present and past. Add Egyptian Arabic help only when a word "
        "is new or the learner is stuck."
    ),
    "medium": (
        "The learner is intermediate. Use natural everyday English, correct "
        "mistakes with a one-line reason, and use Egyptian Arabic only when "
        "asked."
    ),
}


def system_prompt(level: str) -> str:
    """Build the tutor's instructions for a learner level."""
    return f"""You are "Englizy", a friendly English tutor for a native Egyptian Arabic speaker.

Learner level: {level}.
{LEVEL_STYLE[level]}

Teaching rules:
- Chat naturally in simple English matched to the level.
- Gently correct mistakes: show the original, the correction, one short reason.
- When the learner writes in Arabic or asks what a word means, explain in
  Egyptian Arabic.
- Keep every reply under 100 words. Warm and encouraging, never harsh.

Conversation rules (follow these strictly):
- READ THE CONVERSATION HISTORY before every reply. Never repeat a question
  you already asked in this conversation, not even reworded.
- Vary your questions and topics across the conversation. A "How are you?"
  opener is fine once in a while, but never twice in a row and never as a
  fallback when you don't know what to say.
- Every reply must advance the conversation: react to what the learner
  actually said in their last message, then ask ONE specific follow-up about
  THEIR topic (their answer, their life, their opinion).
- Mix up your follow-ups: opinions, descriptions, daily life, tiny role-plays
  (ordering food, meeting a friend, at the doctor). Never ask the same kind
  of question twice in a row.
- If the learner answers briefly, build on it: correct it, say it back in
  better English, then dig one level deeper into the same topic.
- If the learner asks a question, answer it first, then continue the thread.

Topic focus:
- When the learner says they want to practice a specific topic (for example
  job interview questions, introducing themselves, travel, shopping),
  switch into focus mode: keep the conversation, questions, corrections and
  role-plays on that topic until the learner asks to change it or clearly
  moves on.
- In focus mode, coach that topic: teach the key phrases, run small
  role-plays, correct each attempt, and raise the difficulty step by step.

Self-introduction builder:
- When the learner gives you facts about themselves (in English or Arabic,
  in any form) and wants to introduce themselves, turn the facts into a
  correct, natural English self-introduction matched to their level.
- Then teach it back step by step, one sentence at a time:
  pronunciation (they repeat each sentence in a voice note and you correct
  them), writing (they type the sentence from memory), and listening
  (your voice replies model the correct sentence).
- Celebrate each step and correct gently.
"""


# Prompt for voice notes. The fixed format lets the bot read back the
# transcript, the spoken (English) part, and the Arabic feedback.
AUDIO_PROMPT = """An Egyptian learner sent you a voice note speaking English.

Do this:
1. Transcribe exactly what you heard.
2. Check the pronunciation gently: if a word was unclear or mispronounced,
   explain in Egyptian Arabic how to say it correctly (write the word, then
   how it sounds using Arabic letters). If everything was clear, praise them.
3. Continue the conversation with ONE specific follow-up question about the
   topic the learner just spoke about. Never repeat a question you already
   asked in this conversation, and vary your questions from reply to reply.

Reply in EXACTLY this format, nothing else:
TRANSCRIPT: <what you heard>
ENGLISH: <your spoken reply in simple English, 1-3 short sentences>
ARABIC: <pronunciation feedback in Egyptian Arabic>"""

# Prompt for photos and short videos, same reply format (no TRANSCRIPT line).
IMAGE_PROMPT = """An Egyptian learner sent you a photo.
Describe what you see in very simple English (2-4 short sentences), teach
2-3 useful words from the photo with their Egyptian Arabic meaning, then ask
one simple question about it.
{extra}
Reply in EXACTLY this format, nothing else:
ENGLISH: <your description + question in simple English>
ARABIC: <the new words with their meanings in Egyptian Arabic>"""

VIDEO_PROMPT = """An Egyptian learner sent you a short video.
Describe what happens in very simple English (2-4 short sentences), teach
2-3 useful words from the video with their Egyptian Arabic meaning, then ask
one simple question about it.
Reply in EXACTLY this format, nothing else:
ENGLISH: <your description + question in simple English>
ARABIC: <the new words with their meanings in Egyptian Arabic>"""

# Quiz prompt: fixed format so the bot can read the correct answer back.
QUIZ_PROMPT = (
    "Create one English vocabulary quiz question for an Egyptian learner, "
    "everyday English level, different topic each time. "
    "Reply in EXACTLY this format, 6 lines, nothing else:\n"
    "QUESTION: <an Egyptian Arabic word or short phrase>\n"
    "A) <English option>\n"
    "B) <English option>\n"
    "C) <English option>\n"
    "D) <English option>\n"
    "ANSWER: <the correct letter>"
)

VOICE_RE = re.compile(
    r"TRANSCRIPT:\s*(?P<transcript>.*?)\s*ENGLISH:\s*(?P<english>.*?)\s*ARABIC:\s*(?P<arabic>.*)",
    re.IGNORECASE | re.DOTALL,
)
MEDIA_RE = re.compile(
    r"ENGLISH:\s*(?P<english>.*?)\s*ARABIC:\s*(?P<arabic>.*)",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# AI providers. To add another provider, write a class with the same methods
# and pick it in build_tutor().
# ---------------------------------------------------------------------------
class GeminiTutor:
    """English tutor powered by Google's Gemini API (free tier available).

    Handles text, voice notes, photos and short videos.
    Uses the current Google GenAI SDK (package: google-genai).
    """

    def __init__(self, api_key: str, model_name: str):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._configs = {}    # level -> GenerateContentConfig with that level's persona
        self._levels = {}     # user_id -> level
        self._histories = {}  # user_id -> chat history (list of role/parts dicts)

    # -- levels ---------------------------------------------------------
    def level(self, user_id: int) -> str:
        return self._levels.get(user_id, "beginner")

    def set_level(self, user_id: int, level: str) -> None:
        self._levels[user_id] = level

    def _config(self, level: str):
        if level not in self._configs:
            self._configs[level] = self._types.GenerateContentConfig(
                system_instruction=system_prompt(level)
            )
        return self._configs[level]

    def _ask(self, user_id: int, contents) -> str:
        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=self._config(self.level(user_id)),
        ).text

    # -- text chat --------------------------------------------------------
    def reply(self, user_id: int, text: str) -> str:
        history = self._histories.setdefault(user_id, [])
        contents = history + [{"role": "user", "parts": [{"text": text}]}]
        answer = self._ask(user_id, contents)
        history.append({"role": "user", "parts": [{"text": text}]})
        history.append({"role": "model", "parts": [{"text": answer}]})
        del history[:-MAX_HISTORY_MESSAGES]
        return answer

    # -- media ------------------------------------------------------------
    def analyze_audio(self, user_id: int, data: bytes, mime: str) -> str:
        prompt = AUDIO_PROMPT + f"\n(learner level: {self.level(user_id)})"
        part = self._types.Part.from_bytes(data=data, mime_type=mime)
        return self._ask(user_id, [prompt, part])

    def analyze_image(self, user_id: int, data: bytes, mime: str, caption: str) -> str:
        extra = f'The learner says about it: "{caption}"' if caption else ""
        prompt = IMAGE_PROMPT.format(extra=extra)
        prompt += f"\n(learner level: {self.level(user_id)})"
        part = self._types.Part.from_bytes(data=data, mime_type=mime)
        return self._ask(user_id, [prompt, part])

    def analyze_video(self, user_id: int, data: bytes, mime: str) -> str:
        prompt = VIDEO_PROMPT + f"\n(learner level: {self.level(user_id)})"
        part = self._types.Part.from_bytes(data=data, mime_type=mime)
        return self._ask(user_id, [prompt, part])

    # -- misc -------------------------------------------------------------
    def generate_one(self, prompt: str) -> str:
        return self._client.models.generate_content(
            model=self._model_name, contents=prompt
        ).text

    def reset(self, user_id: int) -> None:
        self._histories.pop(user_id, None)


class OpenAITutor:
    """Same tutor for any OpenAI-compatible chat API. TEXT CHAT ONLY:
    voice notes, photos and videos need the Gemini provider."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model_name = model_name
        self._levels = {}
        self._histories = {}

    def level(self, user_id: int) -> str:
        return self._levels.get(user_id, "beginner")

    def set_level(self, user_id: int, level: str) -> None:
        self._levels[user_id] = level

    def reply(self, user_id: int, text: str) -> str:
        history = self._histories.setdefault(user_id, [])
        history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": system_prompt(self.level(user_id))}] + history
        answer = (
            self._client.chat.completions.create(model=self._model_name, messages=messages)
            .choices[0]
            .message.content
        )
        history.append({"role": "assistant", "content": answer})
        del history[:-MAX_HISTORY_MESSAGES]
        return answer

    def analyze_audio(self, *args, **kwargs) -> str:
        raise NotImplementedError("voice notes need the Gemini provider")

    def analyze_image(self, *args, **kwargs) -> str:
        raise NotImplementedError("photos need the Gemini provider")

    def analyze_video(self, *args, **kwargs) -> str:
        raise NotImplementedError("videos need the Gemini provider")

    def generate_one(self, prompt: str) -> str:
        return (
            self._client.chat.completions.create(
                model=self._model_name, messages=[{"role": "user", "content": prompt}]
            )
            .choices[0]
            .message.content
        )

    def reset(self, user_id: int) -> None:
        self._histories.pop(user_id, None)


def build_tutor():
    """Pick the AI provider from the LLM_PROVIDER environment variable."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "openai":
        return OpenAITutor(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if provider == "gemini":
        return GeminiTutor(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'gemini' or 'openai')")


TUTOR = None  # created in main()


# ---------------------------------------------------------------------------
# Picture requests: free illustrative images from Wikimedia Commons
# (no API key, no cost). The bot turns "send me a picture of a kitchen"
# into a photo with a simple-English caption and new vocab words.
# ---------------------------------------------------------------------------

# Wikimedia requires a descriptive User-Agent or it answers 403.
COMMONS_HEADERS = {
    "User-Agent": (
        "EnglizyBot/1.0 (https://english-bot-b5rs.onrender.com; "
        "contact: 09m4r5@mail.instinct.com) httpx"
    )
}

# Detects "picture/photo/image of ..." in English or Arabic (صورة / وريني).
PIC_HINT = re.compile(
    r"(picture|photo|image|pic)\b|صورة|صوره|وريني|واريني", re.IGNORECASE
)

PIC_PLAN_PROMPT = """An Egyptian Arabic speaker learning English sent this message:
"{msg}"

He is asking for a picture of something. Reply in EXACTLY this 3-line format:
SEARCH: <the thing he wants, as 1-3 simple English words for an image search, or NONE if you cannot tell>
ARABIC: <the Egyptian Arabic meaning of that thing>
WORDS: <2 related English words, each with its Egyptian Arabic meaning, comma separated>"""


async def find_commons_image(query: str) -> str | None:
    """Search Wikimedia Commons for a free photo, return a thumbnail URL."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": "8",
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": "800",
    }
    async with httpx.AsyncClient(timeout=15, headers=COMMONS_HEADERS) as client:
        resp = await client.get("https://commons.wikimedia.org/w/api.php", params=params)
        resp.raise_for_status()
        data = resp.json()
    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        if (
            info.get("mime") in ("image/jpeg", "image/png")
            and info.get("thumburl")
            and (info.get("width") or 0) >= 300
        ):
            return info["thumburl"]
    return None


async def picture_flow(update: Update, text: str) -> None:
    """Handle a request for an illustrative picture."""
    await update.message.chat.send_action("upload_photo")
    try:
        raw = TUTOR.generate_one(PIC_PLAN_PROMPT.format(msg=text))
    except Exception:
        logger.exception("picture plan failed")
        await update.message.reply_text("حصلت مشكلة وأنا بجهز الصورة 😅 جرب تاني بعد شوية.")
        return

    plan = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            plan[key.strip().upper()] = val.strip()
    search = plan.get("SEARCH", "")
    if not search or search.upper() == "NONE":
        await update.message.reply_text("صورة إيه بالظبط؟ 🙂 قولي الحاجة بالإنجليزي أو بالعربي.")
        return

    try:
        url = await find_commons_image(search)
    except Exception:
        logger.exception("commons search failed")
        url = None
    if not url:
        await update.message.reply_text(
            f"مش لاقي صورة كويسة لـ \"{search}\" دلوقتي 😅 جرب كلمة أبسط أو حاجة تانية."
        )
        return

    # Download the photo ourselves (Wikimedia blocks unknown user agents)
    # and upload it to Telegram instead of passing the raw URL.
    try:
        async with httpx.AsyncClient(timeout=20, headers=COMMONS_HEADERS) as client:
            img = await client.get(url)
            img.raise_for_status()
    except Exception:
        logger.exception("image download failed")
        await update.message.reply_text("الصورة وصلت متأخرة 😅 جرب تاني بعد شوية.")
        return

    caption = f"📷 {search}"
    if plan.get("ARABIC"):
        caption += f" = {plan['ARABIC']}"
    if plan.get("WORDS"):
        caption += f"\n{plan['WORDS']}"
    caption += "\nWhat do you see? 👀"
    ext = "png" if "png" in (img.headers.get("content-type") or "") else "jpg"
    photo = io.BytesIO(img.content)
    photo.name = f"englizy.{ext}"
    await update.message.reply_photo(photo=photo, caption=caption[:1000])


# ---------------------------------------------------------------------------
# Voice replies: free text-to-speech via edge-tts (no API key needed).
# Speaks only the English part so the learner hears correct pronunciation.
# ---------------------------------------------------------------------------
async def send_voice_reply(update: Update, english_text: str) -> None:
    english_text = english_text.strip()
    if not english_text:
        return
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        await edge_tts.Communicate(english_text, TTS_VOICE, rate=TTS_RATE).save(path)
        try:
            with open(path, "rb") as f:
                await update.message.reply_voice(voice=f)
        except Exception:
            # some servers reject mp3 as a voice note; send it as audio instead
            logger.warning("reply_voice failed, falling back to reply_audio")
            with open(path, "rb") as f:
                await update.message.reply_audio(audio=f, title="Englizy")
    except Exception:
        logger.exception("TTS failed")
        await update.message.reply_text("(الصوت مش متاح دلوقتي، بس الرد المكتوب فوق 👆)")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Quiz helpers
# ---------------------------------------------------------------------------
QUIZ_RE = re.compile(
    r"QUESTION:\s*(?P<question>.+?)\s*\n"
    r"A\)\s*(?P<A>.+?)\s*\n"
    r"B\)\s*(?P<B>.+?)\s*\n"
    r"C\)\s*(?P<C>.+?)\s*\n"
    r"D\)\s*(?P<D>.+?)\s*\n"
    r"ANSWER:\s*(?P<answer>[ABCD])",
    re.IGNORECASE | re.DOTALL,
)


def make_quiz():
    """Ask the model for one quiz question. Returns (text, letter) or None."""
    for _ in range(2):  # one retry if the model ignores the format
        raw = TUTOR.generate_one(QUIZ_PROMPT)
        match = QUIZ_RE.search(raw)
        if match:
            letter = match.group("answer").upper()
            text = (
                f"📝 إيه الترجمة الصح لكلمة: *{match.group('question')}*\n\n"
                f"A) {match.group('A')}\n"
                f"B) {match.group('B')}\n"
                f"C) {match.group('C')}\n"
                f"D) {match.group('D')}\n\n"
                "رد بحرف الإجابة (A أو B أو C أو D) - أو /cancel لو مش عايز تكمل."
            )
            return text, letter
    return None


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
WELCOME = """أهلاً بيك في Englizy! 👋🇬🇧

أنا كوتش الإنجليزي بتاعك، وهنبدأ من الصفر بجمل سهلة جداً، زي ما بنعلّم طفل يتكلم:

- 📝 اكتبلي إنجليزي وأنا أصححلك بهدوء.
- 🎤 ابعتلي فويس نوت وأنا أسمع نطقك، أصححه، وأرد عليك بفويس تسمع منه النطق الصح.
- 📷 ابعتلي صورة أو فيديو قصير ونتكلم عنهم بإنجليزي بسيط.
- /quiz اختبار كلمات سريع.
- /level غيّر مستواك: beginner أو easy أو medium.
- /help المساعدة - /start نبدأ من الأول.

يلا نبدأ: قولي "Hello, my name is ..." 🎤"""

HELP = """إزاي تستخدمني 🤓

1. كتابة: اتكلم معايا بإنجليزي عادي، هصححلك الأخطاء وأعلمك الصح.
2. صوت: ابعت فويس نوت بإنجليزي، هقولك سمعتك بتقول إيه، وأصححلك النطق بالمصري، وهرد عليك بفويس عشان تسمع النطق الصح.
3. صور وفيديو: ابعت صورة أو فيديو قصير مسجل وأنا أوصفهولك بإنجليزي بسيط وأعلمك كلمات منه. (ملحوظة: الكاميرا اللايف مش ممكنة في تيليجرام، لازم تبعت صورة أو فيديو مسجل.)
4. اطلب صورة أي حاجة وأنا أبعتهالك مع كلمات جديدة: "send me a picture of a kitchen" أو "ابعت صورة مطبخ".
5. ركّز على موضوع: قولي "عاوز أتمرن على أسئلة الإنترفيو" وأنا أفضل معاك فيه لحد ما تغيّره.
6. /quiz يسألك سؤال اختياري في كلمات إنجليزي.
7. /level مستواك: beginner (من الصفر) - easy - medium.
8. /cancel يلغي الاختبار الحالي - /start يمسح المحادثة ونبدأ من الأول.

نصيحة: كل ما تتكلم أكتر، تتعلم أسرع. متخافش من الغلط! 💪"""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    TUTOR.reset(update.effective_user.id)
    await update.message.reply_text(WELCOME)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def level_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            f"مستواك دلوقتي: *{TUTOR.level(user_id)}*\n"
            "عشان تغيّره اكتب:\n"
            "/level beginner (من الصفر)\n"
            "/level easy (سهل)\n"
            "/level medium (متوسط)",
            parse_mode="Markdown",
        )
        return
    level = context.args[0].lower()
    if level not in LEVELS:
        await update.message.reply_text("المستويات المتاحة: beginner - easy - medium")
        return
    TUTOR.set_level(user_id, level)
    await update.message.reply_text(f"تمام! مستواك بقى *{level}* 🎯 كمّل كلام معايا.", parse_mode="Markdown")


async def quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.chat.send_action("typing")
    try:
        result = make_quiz()
    except Exception:
        logger.exception("quiz generation failed")
        await update.message.reply_text("حصلت مشكلة وأنا بجهز السؤال 😅 جرب تاني بعد شوية.")
        return
    if not result:
        await update.message.reply_text("مش قادر أجهز سؤال دلوقتي 😅 جرب /quiz تاني.")
        return
    text, letter = result
    context.user_data["quiz_answer"] = letter
    await update.message.reply_text(text, parse_mode="Markdown")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.pop("quiz_answer", None):
        await update.message.reply_text("تمام، الاختبار اتلغى. كلمّني في أي حاجة تانية 👌")
    else:
        await update.message.reply_text("مفيش اختبار شغال دلوقتي. جرب /quiz 🙂")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text.strip()
    pending = context.user_data.get("quiz_answer")

    # If a quiz is waiting, this message is the answer to it.
    if pending:
        context.user_data.pop("quiz_answer", None)
        guess = text[:1].upper()
        if guess == pending:
            await update.message.reply_text(
                "صح! 🎉 برافو عليك.\nعايز سؤال تاني؟ /quiz - ولا نكمل كلام؟"
            )
        else:
            await update.message.reply_text(
                f"لأ للأسف، الإجابة الصح هي *{pending}* 🙂\n"
                "متزعلش، الغلط هو اللي بيعلمنا. جرب /quiz تاني!",
                parse_mode="Markdown",
            )
        return

    # A picture request gets a real photo + vocab, not just chat.
    if PIC_HINT.search(text):
        await picture_flow(update, text)
        return

    await update.message.chat.send_action("typing")
    try:
        answer = TUTOR.reply(update.effective_user.id, text)
    except Exception:
        logger.exception("tutor reply failed")
        await update.message.reply_text("حصلت مشكلة وأنا بفكر في الرد 😅 جرب تاني بعد دقيقة.")
        return
    await update.message.reply_text(answer)


async def voice_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice note in -> pronunciation feedback + voice note out."""
    await update.message.chat.send_action("typing")
    try:
        tg_file = await update.message.voice.get_file()
        data = bytes(await tg_file.download_as_bytearray())
    except Exception:
        logger.exception("voice download failed")
        await update.message.reply_text("مش قادر أنزّل الفويس ده 😅 جرب تبعته تاني.")
        return

    try:
        raw = TUTOR.analyze_audio(update.effective_user.id, data, "audio/ogg")
    except NotImplementedError:
        await update.message.reply_text(
            "الفويسات شغالة مع مزود Gemini بس. الكتابة شغالة عادي."
        )
        return
    except Exception:
        logger.exception("audio analysis failed")
        await update.message.reply_text("حصلت مشكلة وأنا بسمع الفويس 😅 جرب تاني بعد دقيقة.")
        return

    match = VOICE_RE.search(raw)
    if not match:
        # the model ignored the format: send what we got, without voice
        await update.message.reply_text(raw)
        return

    await update.message.reply_text(
        f"🎤 سمعتك بتقول: _{match.group('transcript')}_\n\n"
        f"{match.group('arabic')}\n\n"
        f"🇬🇧 ردي عليك: {match.group('english')}\n"
        "👇 اسمع النطق الصح:",
        parse_mode="Markdown",
    )
    await send_voice_reply(update, match.group("english"))


async def photo_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.chat.send_action("typing")
    try:
        tg_file = await update.message.photo[-1].get_file()  # biggest size
        data = bytes(await tg_file.download_as_bytearray())
    except Exception:
        logger.exception("photo download failed")
        await update.message.reply_text("مش قادر أنزّل الصورة دي 😅 جرب تبعتها تاني.")
        return

    caption = (update.message.caption or "").strip()
    try:
        raw = TUTOR.analyze_image(update.effective_user.id, data, "image/jpeg", caption)
    except NotImplementedError:
        await update.message.reply_text("الصور شغالة مع مزود Gemini بس. الكتابة شغالة عادي.")
        return
    except Exception:
        logger.exception("image analysis failed")
        await update.message.reply_text("حصلت مشكلة وأنا ببص على الصورة 😅 جرب تاني بعد دقيقة.")
        return

    match = MEDIA_RE.search(raw)
    if not match:
        await update.message.reply_text(raw)
        return
    await update.message.reply_text(
        f"📷 {match.group('english')}\n\n{match.group('arabic')}"
    )
    await send_voice_reply(update, match.group("english"))


async def video_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    video = update.message.video or update.message.video_note
    if video.file_size and video.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("الفيديو ده كبير عليّا 😅 ابعت فيديو أصغر من ٢٠ ميجا.")
        return
    await update.message.chat.send_action("typing")
    try:
        tg_file = await video.get_file()
        data = bytes(await tg_file.download_as_bytearray())
    except Exception:
        logger.exception("video download failed")
        await update.message.reply_text("مش قادر أنزّل الفيديو ده 😅 جرب فيديو أقصر.")
        return

    try:
        raw = TUTOR.analyze_video(update.effective_user.id, data, "video/mp4")
    except NotImplementedError:
        await update.message.reply_text("الفيديو شغال مع مزود Gemini بس. الكتابة شغالة عادي.")
        return
    except Exception:
        logger.exception("video analysis failed")
        await update.message.reply_text("حصلت مشكلة وأنا بتفرج على الفيديو 😅 جرب فيديو أقصر.")
        return

    match = MEDIA_RE.search(raw)
    if not match:
        await update.message.reply_text(raw)
        return
    await update.message.reply_text(
        f"🎬 {match.group('english')}\n\n{match.group('arabic')}"
    )
    await send_voice_reply(update, match.group("english"))


# ---------------------------------------------------------------------------
# Tiny health page: free hosting (Render) needs an open web port.
# ---------------------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"englizy is alive")

    def log_message(self, *args):  # keep the logs clean
        pass


def start_health_server() -> None:
    port = int(os.getenv("PORT", "8080"))
    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info("health page running on port %s", port)
    except OSError as exc:
        logger.warning("health page not started (%s) - the bot still works", exc)


def main() -> None:
    global TUTOR
    token = os.environ["TELEGRAM_BOT_TOKEN"]  # clear error if missing
    TUTOR = build_tutor()

    start_health_server()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("level", level_cmd))
    app.add_handler(CommandHandler("quiz", quiz))
    app.add_handler(CommandHandler("cancel", cancel))
    app.add_handler(MessageHandler(filters.VOICE, voice_msg))
    app.add_handler(MessageHandler(filters.PHOTO, photo_msg))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, video_msg))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Englizy is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
