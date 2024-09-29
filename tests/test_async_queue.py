import asyncio

queue = asyncio.Queue()

async def gen():
    while True:
        if not queue.empty():
            v = await queue.get()
            print(f"get value:{v}")
        await asyncio.sleep(0.5)
        continue

async def process():
    for i in range(10000):
        queue.put_nowait(i)
        await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(*[gen(), process()])


if __name__ == "__main__":
    asyncio.run(main())