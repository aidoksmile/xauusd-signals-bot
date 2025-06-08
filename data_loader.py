import yfinance as yf
import pandas as pd
import time
import logging
from typing import Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(
    symbol: str = "GC=F",
    interval_day: str = "1d",
    interval_short: str = "15m",
    period_day: str = "2y",
    period_short: str = "5d",
    max_retries: int = 5,
    retry_delay: int = 5
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Загружает данные по указанному символу на двух таймфреймах.
    """

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[{symbol}] Загрузка данных (попытка {attempt}/{max_retries})")

            df_day = yf.download(tickers=symbol, period=period_day, interval=interval_day, progress=False)
            df_15m = yf.download(tickers=symbol, period=period_short, interval=interval_short, progress=False)

            if not df_day.empty and not df_15m.empty:
                logger.info(f"[{symbol}] Успешно загружено: {len(df_day)} дневных свечей и {len(df_15m)} 15-минутных")
                return df_day, df_15m
            else:
                logger.warning(f"[{symbol}] Получены пустые данные. Пробую снова через {retry_delay} сек...")
                time.sleep(retry_delay)

        except Exception as e:
            logger.error(f"[{symbol}] Ошибка загрузки данных: {e}")
            if attempt < max_retries:
                logger.info(f"Повторная попытка через {retry_delay} сек...")
                time.sleep(retry_delay)
            else:
                logger.critical(f"[{symbol}] Не удалось загрузить данные после {max_retries} попыток.")
                return None, None

    return None, None
