import telegram
import matplotlib.pyplot as plt
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("CHAT_ID")

bot = telegram.Bot(token=bot_token)

def send_signal(signal):
    try:
        msg = f"""
*Сигнал*: {signal['direction']}
*Entry*: {signal['entry']:.2f}
*TP*: {signal['tp']:.2f}
*SL*: {signal['sl']:.2f}
*Risk*: {signal['risk']:.2f}
*Accuracy*: {signal['accuracy']:.2%}
        """
        bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')

        plt.plot(signal['df_15min']['Close'], label='Цена')
        plt.axhline(signal['entry'], color='green', linestyle='--')
        plt.axhline(signal['tp'], color='blue', linestyle='--')
        plt.axhline(signal['sl'], color='red', linestyle='--')
        plt.legend()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            bot.send_photo(chat_id=chat_id, photo=open(tmpfile.name, 'rb'))
            os.unlink(tmpfile.name)
        plt.close()
    except Exception as e:
        print(f"[Telegram] Ошибка отправки: {e}")
