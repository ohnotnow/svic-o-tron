from datetime import datetime, timedelta
from collections import defaultdict

class BotGuard:
    def __init__(self, max_mentions=10, mention_window=timedelta(hours=1)):
        self.mention_counts = defaultdict(list)
        self.max_mentions = max_mentions
        self.mention_window = mention_window

    def should_block(self, message, bot, server_id) -> tuple[bool, bool]:
        """
        Check if a message should be blocked.

        Args:
            message (discord.Message): The message to check.
            bot (discord.Client): The bot instance.
            server_id (str): The ID of the server the bot is running on.

        Returns:
            bool: True if the message should be blocked, False otherwise.
            bool: True if the message should get an abusive reply, False otherwise.
        """
        # ignore DM's
        if message.guild is None:
            return True, False
        # ignore messages not from our our server
        if str(message.guild.id) != server_id:
            return True, False
        # ignore messages from the bot itself
        if message.author == bot.user:
            return True, False
        # ignore messages from other bots
        if message.author.bot:
            return True, False
        # ignore messages without mentions
        if len(message.mentions) == 0:
            return True, False
        # ignore messages where the bot is not mentioned
        if bot.user not in message.mentions:
            return True, False
        # ignore messages without content
        if len(message.content.split(' ', 1)) == 1:
            return True, True

        # keep track of how many times a user has mentioned the bot recently
        user_id = message.author.id
        now = datetime.utcnow()
        self.mention_counts[user_id].append(now)
        self.mention_counts[user_id] = [time for time in self.mention_counts[user_id]
                                        if now - time <= self.mention_window]

        # ignore when the user has mentioned the bot too many times recently
        if len(self.mention_counts[user_id]) > self.max_mentions:
            return True, True

        # ignore when the message doesn't contain regular text (ie only contains mentions, emojis, spaces, etc)
        question = message.content.split(' ', 1)[1][:500].replace('\r', ' ').replace('\n', ' ')
        if not any(char.isalpha() for char in question):
            return True, True

        # all good, allow the message
        return False, False
