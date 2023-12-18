import discord
from dotenv import load_dotenv
import os
import requests
load_dotenv()

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id: return
        
        # print(message.content)
        
        if message.attachments:
            print(message.attachments)

        if message.content.startswith('!tracker'):
            await message.reply('Hello!', mention_author=True)


intents = discord.Intents.default()
intents.message_content = True
print(os.environ['token1'])
client = MyClient(intents=intents)
client.run(os.environ['token1'])
