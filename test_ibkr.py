import asyncio
from ib_insync import IB, util

async def main():
    ib = IB()
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=1)
        print('Connected:', ib.isConnected())
        # fetch account summary
        summary = ib.accountSummary()
        print('Account summary:', summary)
        # disconnect
        ib.disconnect()
    except Exception as e:
        print('Connection error:', e)
    finally:
        if ib.isConnected():
            ib.disconnect()

if __name__ == '__main__':
    util.run(main())