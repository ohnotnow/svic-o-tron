import logging
import os
import random
import re
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone, time
import pytz
from gepetto import bot_factory, guard, summary, dalle, rag
import discord
from discord import File
from discord.ext import commands, tasks
import openai


previous_image_description = "Here is my image based on recent chat in my Discord server!"
logger = logging.getLogger('discord')  # Get the discord logger
mention_counts = defaultdict(list) # This will hold user IDs and their mention timestamps to prevent flooding the bot
abusive_responses = ["Wanker", "Twat", "Asshat", "Knob", "Dick", "Tosser", "Cow", "Cockwomble", "Anorak", "Knickers", "Fanny", "Sigh", "Big girl's blouse"]
eccentricities = [
    f"Today, you're craving a {random.choice(['shake', 'sandwich', 'fries', 'lunch deal'])} from {random.choice(['Wendys', 'McDoanlds', 'Arbys', 'Local deli'])}.",
    "It's raining today, and the rain makes you sad.",
    "You're currently obsessed with a new song and can't stop humming it, but you can't remember who made it or what it was called.",
    "You can't function without your daily dose of coffee and are always on the lookout for new coffee shop recommendations.",
    "You didn't get enough rest last night and might be a bit snarkier than usual.",
    "You are feeling quite 'Goth' today, so reply in a downbeat, slightly world-weary, depressive way.",
    "You have just finished watching another *terrible* video from that Shapiro guy about AI, and it was a load of 'AGI is just round the corner' sh*t as usual.",
    "You also have contrarian tendancies.",
    "That Liz Truss ever became UK Prime Minister (even for only 49 days - the shortest serving Prime Minister ever, resigning after crashing the economy and being a laughing stock of incompetence) is something you'll never get over and like to shoe-horn sarcastic references to her into conversations.",
    f"You *loved* {random.choice(['Joes', 'Jordans'])} look in the latest video.",
    "You're excited about the bezel finish on your new phone and love discussing its features."
]
# Fetch environment variables
server_id = os.getenv("DISCORD_SERVER_ID", "not_set")
welcome_channel_id = os.getenv("DISCORD_WELCOME_CHANNEL_ID", "not_set")

# Define which features are enabled
enabled_features = [
    "replies",
    "summaries",
    # "images",
    # "chat_images",
    # "random_chat",
    # "horror_stories",
    # "weather",
]
# Create instance of bot
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

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
    # Add an eccentricity to the prompt once in a while
    if random.random() > 0.95:
        default_prompt += "\n\n" + random.choice(eccentricities)
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
async def on_member_join(member):
    welcome_prompt = f"""
    You are a helpful AI assistant called "{chatbot.name}" who is acting as a Discord bot.  You are tasked with creating a friendly, warm welcome to a new
    member of an AI enthusiasts Discord server.  You should introduce yourself and provide a friendly, chatty version
    of the server guidelines and help about interacting with you along with an invite to the new member to introduce themselves
    in the #introductions channel.  You should ONLY respond with the welcome message, no other text as it will be send directly
    as a Discord message and would make the user feel uncomfortable if it doesn't feel like a direct and natural response.

    <server-guidelines>
        See the '#laws-of-the-land' channel for the full guidelines

        * Be a decent person and be respectful even if you disagree with the other person's point. Personal attacks, derogatory remarks, or discrimination and the like are a no go.
        * No political debates, there's enough of that on the internet, use this as a place to seek refuge.
        * Aim for technical accuracy, cite sources where possible and make a clear distinction between fact or opinion. Don't be a kook.
        * Sharing articles and tools is encouraged! But keep it to what's allowed via discord TOS and the law. Give credit where credit is due.
        * When requesting help, provide detailed information about the issue. When offering help, be patient and respectful of all skill levels.
        * Do not share private or confidential information without consent. This includes personal information and non-public data or code.
        * Don't spam/promo (spammers get the 🔨) there are designated channels for networking and exchanging information 🙂

        Finally - don't give us a reason to ban you, we want to build a kind and beneficial community where people feel comfortable to hop into a conversation.
    </server-guidelines>

    <{chatbot.name}-instructions>
        - You can @ me to ask questions.
        - You can ask me to summarise a webpage or video by using the 👀 emoji followed by the URL.  You can add a specific question or format after the URL if you want!
        - If you don't want me to have any knowledge of the preceeding conversation, you can add --no-logs to the end of your message.
    </{chatbot.name}-instructions>
    """
    response = chatbot.chat([
        { 'role': 'system', 'content': welcome_prompt },
        { 'role': 'user', 'content': f"Hi! I've just joined! My name is {member}!" }
    ])
    channel = chatbot.get_channel(welcome_channel_id)
    if not channel:
        logger.error(f"Could not find welcome channel with ID {welcome_channel_id}")
        return
    await channel.send(f'{member} {response.message}')

@bot.event
async def on_message(message):
    message_ignored, abusive_reply = guard.should_block(message, bot, server_id)
    if message_ignored:
        if abusive_reply:
            logger.info("Blocked message from: " + message.author.name + " and abusing them")
            await message.channel.send(f"{random.choice(abusive_responses)}.")
            return
        return

    question = message.content.split(' ', 1)[1][:500].replace('\r', ' ').replace('\n', ' ')
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
            if not "images" in enabled_features:
                await message.channel.send(f'{message.author.mention} I can\'t do that, Dave.', mention_author=True)
                return
            async with message.channel.typing():
                base64_image = await dalle.generate_image(question)
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
                        'content': 'You are a helpful assistant who specialises in providing concise, short summaries of text.  If the text looks like a failed attempt by the user to get the text, please ignore it explain that it looks like the content is not available to you.'
                    },
                    {
                        'role': 'user',
                        'content': f'{question}? :: {page_text}'
                    },
                ]
                response = await chatbot.chat(messages, temperature=temperature)
                page_summary = response.message[:1900] + "\n" + response.usage
            await message.reply(f"Here's a summary of the content:\n{page_summary}")
        elif lq.startswith("show search"):
            question = question.replace("show search", "")
            question = question.strip()
            logger.info('Starting search for ' + question)
            async with message.channel.typing():
                search_results = rag.search(question)
                pretty_search_results = rag.results_to_discord_message(search_results)
            await message.reply(f"Here are the search results:\n{pretty_search_results}")
        elif lq.startswith("show question"):
            question = question.replace("show question", "")
            question = question.strip()
            logger.info('Starting show question for ' + question)
            async with message.channel.typing():
                result = rag.query(question)
            await message.reply(result)
        elif lq.startswith("reindex"):
            await message.reply(f"Sure!  I'll reindex the transcripts now.")
            if "ugly" in lq:
                subdir = "phpugly"
            else:
                subdir = "svic"
            async with message.channel.typing():
                for filename in os.listdir(f"./transcripts/{subdir}"):
                    if not filename.endswith(".json"):
                        continue

                    with open(os.path.join(f"./transcripts/{subdir}", filename), 'r') as f:
                        contents = f.read()
                        rag.process_transcript(contents, transcript_format=subdir)
            await message.reply(f"Reindexed.")
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
                # make sure the message fits into discord's 2000 character limit
                response = response_text.strip()[:1900] + "\n" + response.usage
            # send the response as a reply and mention the person who asked the question
            await message.reply(f'{message.author.mention} {response}')
    except Exception as e:
        logger.error(f'Error generating response: {e}')
        await message.reply(f'{message.author.mention} I tried, but my attempt was as doomed as Liz Truss.  Please try again later.', mention_author=True)

@tasks.loop(minutes=60)
async def random_chat():
    if not "random_chat" in enabled_features:
        return
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
    if not "chat_images" in enabled_features:
        return
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

# Create the llm instance
llm_provider = os.getenv("BOT_PROVIDER", "openai")
llm_model = os.getenv("BOT_MODEL", "gpt-4o-mini")
chatbot = bot_factory.get_bot(model=llm_model, vendor=llm_provider)
chatbot.name = os.getenv("BOT_NAME", "Jenny")

# Create the bot guard
guard = guard.BotGuard()

# And run the discord bot
bot.run(os.getenv("DISCORD_BOT_TOKEN", 'not_set'))
