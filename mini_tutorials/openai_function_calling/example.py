import json
import logging
import operator
import sys
import datetime
import openai
import yfinance as yf

TODAY = datetime.date.today().strftime("%Y/%m/%d")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_price(symbol: str, date: str) -> float:
    logger.info(f"Calling get_price with {symbol=} and {date=}")

    history = yf.download(
        symbol, start=date, period="1d", interval="1d", progress=False
    )

    return history["Close"].iloc[0].item()


def calculate(a: float, b: float, op: str) -> float:
    logger.info(f"Calling calculate with {a=}, {b=} and {op=}")

    return getattr(operator, op)(a, b)


get_price_metadata = {
    "name": "get_price",
    "description": "Get closing price of a financial instrument on a given date",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Ticker symbol of a financial instrument",
            },
            "date": {
                "type": "string",
                "description": "Date in the format YYYY-MM-DD",
            },
        },
        "required": ["symbol", "date"],
    },
}

calculate_metadata = {
    "name": "calculate",
    "description": "General purpose calculator",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "First entry",
            },
            "b": {
                "type": "number",
                "description": "Second entry",
            },
            "op": {
                "type": "string",
                "enum": ["mul", "add", "truediv", "sub"],
                "description": "Binary operation",
            },
        },
        "required": ["a", "b", "op"],
    },
}


messages = [
    {"role": "user", "content": sys.argv[1]},
    {
        "role": "system",
        "content": "You are a helpful financial investor who overlooks the "
        f"performance of stocks. Today is {TODAY}. Note that the "
        "format of the date is YYYY/MM/DD",
    },
]

while True:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=messages,
        functions=[get_price_metadata, calculate_metadata],
    )
    message = response["choices"][0]["message"]
    messages.append(message)

    if "function_call" not in message:
        break

    # call custom functions
    function_name = message["function_call"]["name"]
    kwargs = json.loads(message["function_call"]["arguments"])

    if function_name == "get_price":
        output = str(get_price(**kwargs))
    elif function_name == "calculate":
        output = str(calculate(**kwargs))
    else:
        raise ValueError

    messages.append({"role": "function", "name": function_name, "content": output})

print("*" * 80)
print([m["role"] for m in messages])
print("*" * 80)
print(messages[-1]["content"])
