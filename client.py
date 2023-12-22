import discord
from dotenv import load_dotenv
import os
import requests
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import pickle
from tensorflow.keras.models import load_model

# Loading required models and variables needed.
load_dotenv()
model = load_model('kitty_classifier_model.keras')  # Adjust the path accordingly
class_labels = ['cinder', 'stripe', 'both', 'none']

# Function to load and preprocess a single image
def load_and_preprocess_single_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))  # Resize image to match the training size
    img_array = np.array(img) / 255.0  # Normalize pixel values to between 0 and 1
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        print(message)
        # we do not want the bot to reply to itself
        if message.author.id == self.user.id: return
        
        # print(message.content)
        
        if message.attachments:
            for file in message.attachments:
                if not (
                    file.filename.endswith(".jpg")
                    or file.filename.endswith(".jpeg")
                    or file.filename.endswith(".png")
                    or file.filename.endswith(".webp")
                    or file.filename.endswith(".gif")
                ): continue
                
                img_data = requests.get(file.url).content
                with open("image.jpg", "wb") as handler: handler.write(img_data)
            
                image_path = 'image.jpg'  # Replace with your image file path
                new_image = load_and_preprocess_single_image(image_path)

                # Perform inference
                predictions = model.predict(new_image)

                # Decode the prediction
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_labels[predicted_class_index]
                await message.reply(f'That is a picture of {predicted_class}', mention_author=True)

        if message.content.startswith('!tracker'):
            await message.reply('Hello!', mention_author=True)


intents = discord.Intents.default()
intents.message_content = True
print(os.environ['token1'])
client = MyClient(intents=intents)
client.run(os.environ['token1'])
