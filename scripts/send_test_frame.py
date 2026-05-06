import asyncio
import base64
import websockets

WS = 'ws://localhost:8000/ws/teststudent?role=student'

async def main():
    async with websockets.connect(WS) as ws:
        print('Connected to', WS)
        # Wait for CONNECTED message
        msg = await ws.recv()
        print('Recv:', msg)
        # Send candidate info
        await ws.send('{"type":"CANDIDATE_INFO","name":"Test Student","exam_id":"test-exam-1"}')
        print('Sent CANDIDATE_INFO')
        # Send a tiny black JPEG frame as REFERENCE_FACE
        import numpy as np
        import cv2
        img = np.zeros((240,320,3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
        await ws.send('{"type":"REFERENCE_FACE","frame":"' + b64 + '"}')
        print('Sent REFERENCE_FACE')
        # Send one FRAME
        await ws.send('{"type":"FRAME","frame":"' + b64 + '"}')
        print('Sent FRAME')
        # Read a few responses (non-blocking with timeout)
        for i in range(5):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print('Recv:', msg)
            except asyncio.TimeoutError:
                print('No more messages, exiting')
                break

if __name__ == '__main__':
    asyncio.run(main())
