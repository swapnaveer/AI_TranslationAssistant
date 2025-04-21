import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
from bleurt import score as bleurt_score
import tensorflow as tf
import openai
import pyttsx3
from dotenv import load_dotenv
import os

load_dotenv()
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# BLEURT setup
bleurt_checkpoint = "bleurt/BLEURT-20"
scorer = bleurt_score.BleurtScorer(bleurt_checkpoint)

# Load translation models
MODEL_NAME_MAPPING = {
    ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
    ("English", "German"): "Helsinki-NLP/opus-mt-en-de",
    ("English", "Hindi"): "Helsinki-NLP/opus-mt-en-hi",
    ("English", "Telugu"): "Helsinki-NLP/opus-mt-en-te",
    ("English", "Tamil"): "Helsinki-NLP/opus-mt-en-ta",
    # Add other combinations as needed
}

LANGUAGES = ["English", "French", "Spanish", "German", "Hindi", "Telugu", "Tamil"]

# Setup text-to-speech
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_translation_model(src_lang, tgt_lang):
    model_name = MODEL_NAME_MAPPING.get((src_lang, tgt_lang))
    if not model_name:
        raise ValueError("Unsupported language pair")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, src_lang, tgt_lang):
    model, tokenizer = get_translation_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

def score_legend():
    return """🔍 **BLEURT Score Legend**
    - **0.80 – 1.00**: Excellent translation
    - **0.60 – 0.79**: Good, minor issues
    - **0.40 – 0.59**: Fair, some errors
    - **Below 0.40**: Poor quality
    """

def gpt_post_process(translation, src_lang, tgt_lang):
    if not openai.api_key:
        return translation + " (GPT enhancement disabled)"

    try:
        prompt = (
                f"The following text is a machine-translated sentence from {src_lang} to {tgt_lang}. "
                f"Please improve its grammar and fluency without changing its meaning:\n\n{translation}\n\n"
                "Return only the improved sentence."
            )
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant. Improve the following {src_lang} to English translation to make it more natural, accurate, and contextually appropriate.",
                    },
                    {
                        "role": "user",
                        "content": translation,
                    },
                ],
                temperature=0.3,
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        return translation + " (GPT skipped: quota exceeded)"
    except Exception as e:
        return translation + f" (GPT skipped: {str(e)})"

def get_bleurt_score(ref, candidate):
    return scorer.score(references=[ref], candidates=[candidate])[0]

def translate(input_text, src_lang, tgt_lang, use_gpt):
    translated_text = translate_text(input_text, src_lang, tgt_lang)
    enhanced_text = gpt_post_process(translated_text, src_lang, tgt_lang) if use_gpt else translated_text

    bleurt_original = round(get_bleurt_score(input_text, translated_text), 3)
    bleurt_enhanced = round(get_bleurt_score(input_text, enhanced_text), 3) if use_gpt else None

    return {
        "Original Translation": translated_text,
        "Enhanced Translation": enhanced_text if use_gpt else "(Enhancement not applied)",
        "BLEURT Score (Original)": bleurt_original,
        "BLEURT Score (Enhanced)": bleurt_enhanced if use_gpt else "(N/A)",
    }

def app():
    with gr.Blocks() as demo:
        gr.Markdown("## 🌍 AI Localization Enhancer with BLEURT & GPT")

        with gr.Row():
            src_lang = gr.Dropdown(label="Source Language", choices=LANGUAGES, value="English")
            tgt_lang = gr.Dropdown(label="Target Language", choices=LANGUAGES, value="French")

        with gr.Accordion("BLEURT Score Legend", open=False):
            gr.Markdown(score_legend())

        input_text = gr.Textbox(label="Enter text to translate", lines=3)
        use_gpt = gr.Checkbox(label="Enhance translation with GPT", value=True)

        translate_btn = gr.Button("Translate")
        output_original = gr.Textbox(label="Original Translation")
        output_enhanced = gr.Textbox(label="Enhanced Translation")
        output_bleurt_orig = gr.Textbox(label="BLEURT Score (Original)")
        output_bleurt_enhanced = gr.Textbox(label="BLEURT Score (Enhanced)")
        tts_button = gr.Button("🔊 Listen to Translation")

        def run_translation(text, src, tgt, enhance):
            result = translate(text, src, tgt, enhance)
            return result["Original Translation"], result["Enhanced Translation"], result["BLEURT Score (Original)"], result["BLEURT Score (Enhanced)"]

        def run_tts(text):
            speak(text)
            return "Playing..."

        translate_btn.click(run_translation, [input_text, src_lang, tgt_lang, use_gpt], [output_original, output_enhanced, output_bleurt_orig, output_bleurt_enhanced])
        tts_button.click(run_tts, output_enhanced, None)

    demo.launch()

if __name__ == "__main__":
    app()
