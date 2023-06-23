import asyncio
import logging

import httpx

logger = logging.getLogger()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(format="%(asctime)s %(name)s %(message)s", level=logging.INFO)


async def send_request(client: httpx.AsyncClient, semaphore: asyncio.Semaphore) -> int:
    url = "https://pokeapi.co/api/v2/pokemon/ditto"
    async with semaphore:
        logger.info("Sending request")
        response = await client.get(url)
        logger.info("Response received")

    return response.status_code


async def main() -> int:
    semaphore = asyncio.Semaphore(5)
    async with httpx.AsyncClient() as client:
        tasks = [asyncio.create_task(send_request(client, semaphore)) for _ in range(10)]
        status_codes = await asyncio.gather(*tasks)

    logger.info("All work done")

    return 0 if all(c == 200 for c in status_codes) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
