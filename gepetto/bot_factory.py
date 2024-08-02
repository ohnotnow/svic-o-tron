from gepetto import anyscale, gpt, ollama, groq, claude

def get_bot(model="gpt-4o", vendor="unknown"):
    if model.startswith('gpt'):
        bot = gpt.GPTModel(model=model)
    elif model.startswith('claude'):
        bot = claude.ClaudeModel(model=model)
    elif vendor.startswith('ollama'):
        bot = ollama.OllamaModel(model=model)
    elif vendor.startswith('groq'):
        bot = groq.GroqModel(model=model)
    elif vendor.startswith('anyscale'):
        bot = anyscale.MistralModel(model=model)
    else:
        raise ValueError(f"Cannot find a bot for : {model} / {vendor}")
    return bot
