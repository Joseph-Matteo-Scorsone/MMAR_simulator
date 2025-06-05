from quart import Quart, render_template, websocket
import asyncio
import numpy as np
from simulation import multifractal_returns

app = Quart(__name__)

# WebSocket enpoint
@app.websocket('/ws')
async def ws():
    try:
        n = 10
        H = 0.6
        m = 0.6
        np.random.seed(42)

        theta, returns = multifractal_returns(n, H, m)

        idx = 0
        total = len(theta)

        while idx < total:
            data = {
                "time": theta[:idx+1].tolist(),
                "returns": returns[:idx+1].tolist(),
            }

            await websocket.send_json(data)
            idx += 1

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"WebSocket error: {e}")


@app.route('/')
async def index():
    return await render_template('index.html')

if __name__ == "__main__":
    app.run('localhost', 8000)
