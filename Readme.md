# Discord LLM Bot

This bot uses LLMs to generate responses to messages in a Discord server. It has some special functionality to track the number of times a user has mentioned the bot recently and limit the number of LLM calls.

It has a couple of extra options to do common things.  Eg:
```
@Gepetto create an image of a rocket flying through space
@Gepetto 👀 https://www.example.com/an/article
@Gepetto 👀 <https://www.youtube.com/watch?v=123f830q9>
@Gepetto 👀 <https://www.example.com/an/article> can you give me the main insights on this as bullet points?
```
The youtube one depends on their being subtitles/transcripts attached to the video.  The summarise command is a little limited (currently hard-coded) in scope due to token limits on different models so a 'lowest common denominator' value is used.

## Environment Variables

The following environment variables are used:

- `DISCORD_BOT_TOKEN`: Your Discord bot token
- `DISCORD_SERVER_ID`: The ID of your Discord server
- `BOT_PROVIDER`: The LLM API provider you're using (can be 'groq', 'openai', 'claude')
- `BOT_MODEL`: The particular model to use by default (eg, 'gpt-4o-mini')
- `BOT_NAME`: The name of the bot (single word)
- `DISCORD_BOT_DEFAULT_PROMPT`: Your default system prompt (optional)

Depending which LLM backend you are using you will also need one of :

- `OPENAI_API_KEY`: Your OpenAI API key
- `GROQ_API_KEY`: If using groq
- `CLAUDE_API_KEY`: if using anthropoic/claude

## Cloning the repo

```sh
git clone https://github.com/ohnotnow/svic-o-tron.git
cd svic-o-tron
```

## Running the Script manually

To run the bot:

1. Install the required Python dependencies.  You can install these by running `pip install -r requirements.txt`.
2. Set your environment variables. These can be set in your shell, or stored in a `.env` file at the root of your project.
3. Run `python main.py` in the root of your project to start the bot.

## Docker Deployment

A `Dockerfile` is included in this repository. This allows you to easily build and run your bot inside a Docker container.  Please see the `run.sh` script for an example of building and running the container.  The `run.sh` version assumes it will be used to run multiple different bots (different system prompts or LLMs).  So for each
you would create a `.env.name-of-bot` file with the specific environment variables for that version - then run `run.sh name-of-bot` to use that env.

## License

Released under the MIT License.  See [LICENSE.md] for details.
