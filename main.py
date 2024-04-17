import logging
import os
import random
import re
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone, time
import pytz
from gepetto import mistral, dalle, summary, gpt, stats, groq, claude, ollama
import discord
from discord import File
from discord.ext import commands, tasks
import openai


previous_image_description = "Here is my image based on recent chat in my Discord server!"
logger = logging.getLogger('discord')  # Get the discord logger
mention_counts = defaultdict(list) # This will hold user IDs and their mention timestamps to prevent flooding the bot
abusive_responses = ["Wanker", "Asshole", "Prick", "Twat", "Asshat", "Knob", "Dick", "Tosser", "Cow", "Cockwomble", "Anorak", "Knickers", "Fanny", "Sigh", "Big girl's blouse"]

# Fetch environment variables
server_id = os.getenv("DISCORD_SERVER_ID", "not_set")
model_engine = os.getenv("OPENAI_MODEL_ENGINE", gpt.Model.GPT3_5_Turbo.value[0])
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create instance of bot
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def get_chatbot():
    chatbot = None
    if os.getenv("BOT_PROVIDER") == 'mistral':
        chatbot = mistral.MistralModel()
    elif os.getenv("BOT_PROVIDER") == 'groq':
        chatbot = groq.GroqModel()
    elif os.getenv("BOT_PROVIDER") == 'claude':
        chatbot = claude.ClaudeModel()
    elif os.getenv("BOT_PROVIDER") == 'ollama':
        chatbot = ollama.OllamaModel()
    else:
        chatbot = gpt.GPTModel()
    return chatbot

def remove_nsfw_words(message):
    message = re.sub(r"(fuck|prick|asshole|shit|wanker|dick)", "", message)
    return message

async def get_history_as_openai_messages(channel, include_bot_messages=True, limit=10, since_hours=None, nsfw_filter=False):
    messages = []
    total_length = 0
    if since_hours:
        after_time = datetime.utcnow() - timedelta(hours=since_hours)
    else:
        after_time = None
    async for msg in channel.history(limit=limit, after=after_time):
        # bail out if the message was by a bot and we don't want bot messages included
        if (not include_bot_messages) and (msg.author.bot):
            continue
        # The role is 'assistant' if the author is the bot, 'user' otherwise
        role = 'assistant' if msg.author == bot.user else 'user'
        username = "" if msg.author == bot.user else msg.author.name
        content = remove_emoji(msg.content)
        message_content = f"{content}"
        message_content = re.sub(r'\[tokens used.+Estimated cost.+]', '', message_content, flags=re.MULTILINE)
        message_content = remove_nsfw_words(message_content) if nsfw_filter else message_content
        message_length = len(message_content)
        if total_length + message_length > 1000:
            break
        messages.append({
            "role": role,
            "content": message_content,
        })
        total_length += message_length
    messages = messages[1:]  # Exclude the mention message
    # We reverse the list to make it in chronological order
    return messages[::-1]

def build_messages(question, extended_messages, system_prompt=None):
    now = datetime.now()
    day = now.strftime("%d")
    suffix = lambda day: "th" if 11 <= int(day) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(int(day) % 10, "th")
    formatted_date = now.strftime("%B %d" + suffix(day) + ", %Y %I:%M %p")
    if system_prompt is None:
        default_prompt = os.getenv('DISCORD_BOT_DEFAULT_PROMPT', f'You are a helpful AI assistant called "{chatbot.name}" who specialises in providing answers to questions.  You should ONLY respond with the answer, no other text.')
    else:
        default_prompt = system_prompt
    extended_messages.append(
        {
            'role': 'user',
            'content': f'{question}'
        },
    )
    extended_messages.append(
        {
            'role': 'system',
            'content': f'Today is {formatted_date}. {default_prompt}.'
        }
    )

    return extended_messages

def remove_emoji(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

@bot.event
async def on_ready():
    random_chat.start()
    make_chat_image.start()
    logger.info(f"Using model type : {type(chatbot)}")

@bot.event
async def on_message(message):
    # ignore direct messages
    if message.guild is None:
        return

    # Ignore messages not sent by our server
    if str(message.guild.id) != server_id:
        return

    # Ignore messages sent by the bot itself
    if message.author == bot.user:
        return

    # Ignore messages that don't mention anyone at all
    if len(message.mentions) == 0:
        return

    # Bail out if the bot was not mentioned
    if not (bot.user in message.mentions):
        return

    user_id = message.author.id
    username = message.author.name
    logger.info(f'Bot was mentioned by user {username} (ID: {user_id})')

    # If the user is a bot then send an abusive response
    if message.author.bot:
        await message.channel.send(f"{random.choice(abusive_responses)}.")
        return

    # Current time
    now = datetime.utcnow()

    # Add the current time to the user's list of mention timestamps
    mention_counts[user_id].append(now)
    # Remove mentions that were more than an hour ago
    mention_counts[user_id] = [time for time in mention_counts[user_id] if now - time <= timedelta(hours=1)]
    # If the user has mentioned the bot more than 10 times recently
    if len(mention_counts[user_id]) > 10:
        # Send an abusive response
        await message.reply(f"{message.author.mention} {random.choice(abusive_responses)}.")
        return

    # If the message is just a mention, send an abusive response
    if len(message.content.split(' ', 1)) == 1:
        await message.reply(f"{message.author.mention} {random.choice(abusive_responses)}.")
        return

    # If the message is just punctuation or whatever, send an abusive response
    question = message.content.split(' ', 1)[1][:500].replace('\r', ' ').replace('\n', ' ')
    logger.info(f'Question: {question}')
    if not any(char.isalpha() for char in question):
        await message.channel.send(f'{message.author.mention} {random.choice(abusive_responses)}.')
        return

    if "--strict" in question.lower():
        question = question.lower().replace("--strict", "")
        temperature = 0.1
    elif "--wild" in question.lower():
        question = question.lower().replace("--wild", "")
        temperature = 1.5
    elif "--tripping" in question.lower():
        question = question.lower().replace("--tripping", "")
        temperature = 1.9
    else:
        temperature = 1.0

    # pattern = r"summarise\s+(<)?http"
    pattern = r"👀\s*\<?(http|https):"

    try:
        lq = question.lower().strip()
        if lq.startswith("create an image") or lq.startswith("📷") or lq.startswith("🖌️") or lq.startswith("🖼️"):
            async with message.channel.typing():
                base64_image = await dalle.generate_image(question)
            stats.update(message.author.id, message.author.name, 0, 0.04)
            await message.reply(f'{message.author.mention}\n_[Estimated cost: US$0.04]_', file=base64_image, mention_author=True)
        elif re.search(pattern, lq):
            question = question.replace("👀", "")
            question = question.strip()
            question = question.strip("<>")
            page_summary = ""
            async with message.channel.typing():
                page_text = await summary.get_text(message, question.strip())
                messages = [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant who specialises in providing concise, short summaries of text.'
                    },
                    {
                        'role': 'user',
                        'content': f'{question}? :: {page_text}'
                    },
                ]
                response = await chatbot.chat(messages, temperature=1.0)
                stats.update(message.author.id, message.author.name, response.tokens, response.cost)
                page_summary = response.message[:1900] + "\n" + response.usage
            await message.reply(f"Here's a summary of the content:\n{page_summary}")
        elif question.lower().strip() == "stats":
            statistics = stats.get_stats()
            response = f"```json\n{json.dumps(statistics, indent=2)}\n```"
            await message.reply(f'{message.author.mention} {response}', mention_author=True)
        else:
            async with message.channel.typing():
                if "--no-logs" in question.lower():
                    context = []
                    question = question.lower().replace("--no-logs", "")
                else:
                    if chatbot.uses_logs:
                        context = await get_history_as_openai_messages(message.channel)
                    else:
                        context = []
                messages = build_messages(question, context)
                response = await chatbot.chat(messages, temperature=temperature)
                response_text = response.message
                # try and remove LLM 'added extras'
                response_text = re.sub(r'\[tokens used.+Estimated cost.+]', '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r"Gepetto' said: ", '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r"Minxie' said: ", '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r"^.*At \d{4}-\d{2}.+said?", "", response_text, flags=re.MULTILINE)
                stats.update(message.author.id, message.author.name, response.tokens, response.cost)
                # make sure the message fits into discord's 2000 character limit
                response = response_text.strip()[:1900] + "\n" + response.usage
            # send the response as a reply and mention the person who asked the question
            await message.reply(f'{message.author.mention} {response}')
    except Exception as e:
        logger.error(f'Error generating response: {e}')
        await message.reply(f'{message.author.mention} I tried, but my attempt was as doomed as Liz Truss.  Please try again later.', mention_author=True)

@tasks.loop(minutes=60)
async def random_chat():
    logger.info("In random_chat")
    if not isinstance(chatbot, gpt.GPTModel):
        logger.info("Not joining in with chat because we are using non-gpt")
        return
    if random.random() > 0.3:
        logger.info("Not joining in with chat because random number is too high")
        return
    now = datetime.now().time()
    start = datetime.strptime('23:00:00', '%H:%M:%S').time()
    end = datetime.strptime('07:00:00', '%H:%M:%S').time()
    if (now >= start or now <= end):
        logger.info("Not joining in with chat because it is night time")
        return
    channel = bot.get_channel(int(os.getenv('DISCORD_BOT_CHANNEL_ID', 'Invalid').strip()))
    context = await get_history_as_openai_messages(channel, include_bot_messages=False, since_hours=0.5)
    if len(context) < 5:
        logger.info("Not joining in with chat because it is too quiet")
        return
    system_prompt = f'You are a helpful AI Discord bot called "{chatbot.name}" who reads the chat history of a Discord server and adds funny, acerbic, sarcastic replies based on a single topic mentioned.  Your reply should be natural and fit in with the flow of the conversation as if you were a human user chatting to your friends on Discord.  You should ONLY respond with the chat reply, no other text.  You can quote the text you are using as context by using markdown `> original text here` formatting for context but do not @mention the user.'
    context.append(
        {
            'role': 'system',
            'content': system_prompt
        }
    )
    response = await chatbot.chat(context, temperature=1.0)
    await channel.send(f"{response.message[:1900]}\n{response.usage}")

@tasks.loop(time=time(hour=17, tzinfo=pytz.timezone('Europe/London')))
async def make_chat_image():
    global previous_image_description
    logger.info("In make_chat_image")
    if not isinstance(chatbot, gpt.GPTModel):
        logger.info("Not saying something random because we are not using GPT")
        return
    channel = bot.get_channel(int(os.getenv('DISCORD_BOT_CHANNEL_ID', 'Invalid').strip()))
    async with channel.typing():
        history = await get_history_as_openai_messages(channel, limit=50, nsfw_filter=True)
        combined_chat = "Could you make me an image which takes only one or two of the themes contained in following transcript? Don't try and cover too many things in one image. Please make the image an artistic interpretation - not a literal image based on the summary. Be creative! Choose a single artistic movement from across the visual arts, historic or modern. The transcript is between adults - so if there has been any NSFW content or mentions of celebtrities, please just make an image a little like them but not *of* them.  Thanks!\n\n"
        for message in history:
            combined_chat += f"{message['content']}\n"
        discord_file, prompt = await dalle.generate_image(combined_chat, return_prompt=True)
        if discord_file is None:
            await channel.send(f"Sorry, I tried to make an image but I failed (probably because of naughty words - tsk).")
            return
        try:
            response = await chatbot.chat([{
                'role': 'user',
                'content': f"Could you reword the following sentence to make it sound more like a jaded, cynical human who works as a programmer wrote it? You can reword and restructure it any way you like - just keep the sentiment and tone. <sentence>{previous_image_description}</sentence>.  Please reply with only the reworded sentence as it will be sent directly to Discord as a message."
            }])
        except Exception as e:
            logger.error(f'Error generating chat image response: {e}')
            await channel.send(f"Sorry, I tried to make an image but I failed (probably because of naughty words - tsk).")
            return
    previous_image_description = response.message
    await channel.send(f'{response.message}\n> {prompt}\n_[Estimated cost: US$0.05]_', file=discord_file)

# Run the bot
chatbot = get_chatbot()
bot.run(os.getenv("DISCORD_BOT_TOKEN", 'not_set'))
