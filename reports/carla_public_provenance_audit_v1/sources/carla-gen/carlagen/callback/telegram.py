"""
Miscellaneous callbacks

"""

from typing import TYPE_CHECKING

from .callback import Callback
import asyncio

# optional telegram import
try:
    import telegram
except ImportError:
    telegram = None

if TYPE_CHECKING:
    from carlagen import Simulation


class SendTelegramMessageCallback(Callback):
    """
    Send a telegram message if the simulation is started and when it is finished
    """

    def __init__(self, token, chat_id, message):
        super().__init__()
        self.token = token
        self.chat_id = chat_id
        self.message = message

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        bot = telegram.Bot(token=self.token)

        async def send_message():
            await bot.send_message(chat_id=self.chat_id, text=self.message)

        asyncio.run(send_message())

        self.log.info(f"Sent end message to {self.chat_id}")
