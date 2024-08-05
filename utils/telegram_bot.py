import asyncio
import threading

from utils.singleton import Singleton
from constants import USE_TELEGRAM, CHAT_ID, USE_TELEGRAM
from telegram import Bot
from telegram.error import TelegramError

class TelegramBot(metaclass = Singleton):
    def __init__(self, token):
        if not USE_TELEGRAM:
            self.bot = None
            return
        self.bot = Bot(token=token)

    async def send_message(self, text, chat_id=CHAT_ID, ):
        if not USE_TELEGRAM:
            return
        try:
            await self.bot.send_message(chat_id=chat_id, text=text)
        except TelegramError as e:
            print(f"Failed to send message: {e}")

    async def send_photo(self, photo, chat_id=CHAT_ID, caption=None):
        if not USE_TELEGRAM:
            return
        try:
            await self.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
        except TelegramError as e:
            print(f"Failed to send photo: {e}")

def update_telegram(bot, message):
    asyncio.run(bot.send_message(message))

async def main():
    bot_token = 'YOUR_BOT_TOKEN'
    chat_id = 'YOUR_CHAT_ID'
    message = 'Ciao, questo è un messaggio inviato dal mio bot Telegram!'

    bot = TelegramBot(bot_token)
    await bot.send_message(message, chat_id)

if __name__ == "__main__":
    asyncio.run(main())
