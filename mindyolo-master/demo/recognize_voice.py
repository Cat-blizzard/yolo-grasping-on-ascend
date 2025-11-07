# coding: utf-8
import _thread as thread
import time
import base64
import datetime
import hashlib
import hmac
import json
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import pyaudio
import ssl

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1, "vad_eos":10000}

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'), digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        url = url + '?' + urlencode(v)
        return url

class ASRRecognizer:
    def __init__(self, appid, apikey, apisecret, on_result=None, max_duration=10, intervel=0.04):
        self.wsParam = Ws_Param(appid, apikey, apisecret)
        self.on_result = on_result
        self.max_duration = max_duration
        self.intervel = intervel
        self.final_result = ""
        self.ws = None

    def _on_message(self, ws, message):
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            if code != 0:
                errMsg = json.loads(message)["message"]
                print(f"sid:{sid} call error:{errMsg} code is:{code}")
            else:
                data = json.loads(message)["data"]["result"]["ws"]
                result = ""
                for i in data:
                    for w in i["cw"]:
                        result += w["w"]
                self.final_result += result
                if self.on_result:
                    self.on_result(result)
        except Exception as e:
            print("receive msg,but parse exception:", e)

    def _on_error(self, ws, error):
        print("### error:", error)

    def _on_close(self, ws, a, b):
        print("### closed ###")
        print("整句识别结果: ", self.final_result)

    def _on_open(self, ws):
        def run(*args):
            frameSize = 8000
            status = STATUS_FIRST_FRAME
            start_time = time.time()
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=frameSize)
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.max_duration:
                    status = STATUS_LAST_FRAME
                buf = stream.read(frameSize)
                if status == STATUS_FIRST_FRAME:
                    d = {"common": self.wsParam.CommonArgs,
                         "business": self.wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                elif status == STATUS_CONTINUE_FRAME:
                    d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                time.sleep(self.intervel)
            stream.stop_stream()
            stream.close()
            p.terminate()
            ws.close()
        thread.start_new_thread(run, ())

    def recognize(self):
        wsUrl = self.wsParam.create_url()
        self.ws = websocket.WebSocketApp(wsUrl,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        self.ws.on_open = self._on_open
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def recognize_once(self):
        """
        阻塞式识别，返回整句识别结果。
        """
        self.final_result = ""
        wsUrl = self.wsParam.create_url()
        done = []
        def on_result(result):
            # 可选：实时回调
            if self.on_result:
                self.on_result(result)
        def on_close(ws, a, b):
            done.append(True)
        ws = websocket.WebSocketApp(wsUrl,
                                    on_message=self._on_message,
                                    on_error=self._on_error,
                                    on_close=on_close)
        ws.on_open = self._on_open
        import threading
        t = threading.Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        t.start()
        while not done:
            time.sleep(0.1)
        t.join()
        return self.final_result

# 仅用于脚本直接运行时测试
# if __name__ == "__main__":
#     def print_partial_result(result):
#         print("实时识别结果:", result)
#     recognizer = ASRRecognizer(
#         appid='3cdb2c75',
#         apikey='52895af0e672d6d172d478de4e027b07',
#         apisecret='ZGRiNDAyMmVmYWQ0YjVhOTAyNDYzMzcw',
#         on_result=print_partial_result,
#         max_duration=10,
#         intervel=0.04
#     )
#     recognizer.recognize()
def asr_recognize(
    max_duration: float = 10.0,
    interval_sec: float = 0.04,
    on_partial=None,
) -> str:
    """
    对外统一接口：启动一次语音识别并返回整句结果（str）。
    - appid/apikey/apisecret: 科大讯飞控制台获取
    - max_duration: 录音时长上限（秒）
    - interval_sec: 推流分帧间隔（秒），需与类里保持一致
    - on_partial: 可选的增量回调函数，形如 on_partial(text:str) -> None
    """
    rec = ASRRecognizer(
        appid='3cdb2c75',
        apikey='52895af0e672d6d172d478de4e027b07',
        apisecret='ZGRiNDAyMmVmYWQ0YjVhOTAyNDYzMzcw',
        on_result=on_partial,
        max_duration=max_duration,
        intervel=interval_sec,
    )
    # 阻塞直至识别完成，返回整句文本
    return rec.recognize_once()
