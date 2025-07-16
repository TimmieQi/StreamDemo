# server.py (H.265 + RTP over UDP 重构版)

import io
import cv2
import socket
import time
import threading
import json
import os
import av
import pyaudio
import numpy as np

from shared_config import *

# --- 视频文件扫描 (无变化) ---
def get_video_files(path="videos"):
    if not os.path.exists(path):
        os.makedirs(path)
        return []
    supported_formats = ('.mp4', '.mkv', '.avi', '.mov')
    return [f for f in os.listdir(path) if f.endswith(supported_formats)]

# --- 自适应流控器 (已更新) ---
class AdaptiveStreamController:
    def __init__(self):
        # [MODIFIED] 策略现在控制比特率，而不是JPEG质量
        self.configs = {
            "good": {"resolution": (640, 480), "bitrate": 1000 * 1024, "fps_limit": 30},
            "medium": {"resolution": (480, 360), "bitrate": 500 * 1024, "fps_limit": 20},
            "poor": {"resolution": (320, 240), "bitrate": 250 * 1024, "fps_limit": 15}
        }
        self.current_strategy_name = "good"
        self.lock = threading.Lock()

    def get_current_strategy(self):
        with self.lock:
            return self.configs[self.current_strategy_name]

    def update_strategy(self, loss_rate):
        new_strategy_name = "good"
        if loss_rate >= 0.1:
            new_strategy_name = "poor"
        elif loss_rate >= 0.03:
            new_strategy_name = "medium"

        with self.lock:
            if self.current_strategy_name != new_strategy_name:
                print(f"[服务端-控制器] 丢包率: {loss_rate:.2%}, 切换策略至: {new_strategy_name.upper()}")
                self.current_strategy_name = new_strategy_name

# --- 推流管理器 (无变化) ---
class StreamerManager:
    def __init__(self, video_sock, audio_sock, controller):
        self.video_sock = video_sock
        self.audio_sock = audio_sock
        self.controller = controller
        self.current_stream_thread = None
        self.stream_control = {'running': False, 'seek_to': -1.0}
        self.lock = threading.Lock()

    def start_stream(self, source, client_addr):
        with self.lock:
            print("[服务端-管理器] 请求开启新推流...")
            self.stop_stream()
            self.stream_control = {'running': True, 'seek_to': -1.0}

            if source == "camera":
                print("[服务端-管理器] 启动摄像头直播...")
                self.current_stream_thread = threading.Thread(
                    target=stream_from_camera,
                    args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr),
                    daemon=True
                )
                self.current_stream_thread.start()
                return {"duration": 0}
            else:
                video_path = os.path.join("videos", source)
                if os.path.exists(video_path):
                    print(f"[服务端-管理器] 启动文件点播: {source}")
                    try:
                        with av.open(video_path) as container:
                            duration_sec = container.duration / av.time_base if container.duration else 0
                    except Exception as e:
                        print(f"[服务端-管理器] 错误: 无法读取视频文件信息 {video_path}: {e}")
                        return None

                    self.current_stream_thread = threading.Thread(
                        target=stream_from_file,
                        args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr, video_path),
                        daemon=True
                    )
                    self.current_stream_thread.start()
                    return {"duration": duration_sec}
                else:
                    print(f"[服务端-管理器] 错误: 找不到视频文件 {video_path}")
                    return None

    def stop_stream(self):
        if self.stream_control['running']:
            self.stream_control['running'] = False
            print("[服务端-管理器] 发送停止信号到当前推流线程...")

        if self.current_stream_thread and self.current_stream_thread.is_alive():
            self.current_stream_thread.join(timeout=1.0)

        self.current_stream_thread = None
        print("[服务端-管理器] 推流已确认停止。")

    def seek_stream(self, target_time_sec):
        with self.lock:
            if self.stream_control['running']:
                print(f"[服务端-管理器] 请求跳转到 {target_time_sec:.2f} 秒")
                self.stream_control['seek_to'] = target_time_sec

# --- 控制信道处理 (无变化) ---
def control_channel_handler(sock, manager):
    client_info = {'addr': None, 'last_contact': 0}
    video_files = get_video_files()

    def watchdog():
        while True:
            time.sleep(5)
            if client_info['addr'] and (time.time() - client_info['last_contact'] > 5):
                print(f"[服务端-看门狗] 客户端 {client_info['addr']} 超时，停止推流。")
                manager.stop_stream()
                client_info['addr'] = None

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[服务端-控制] 可用视频文件: {video_files}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            client_info.update({'addr': addr, 'last_contact': time.time()})
            message = json.loads(data.decode())
            command = message.get("command")

            if command == "get_list":
                sock.sendto(json.dumps(["camera"] + video_files).encode(), addr)
            elif command == "play":
                play_info = manager.start_stream(message.get("source"), addr)
                if play_info:
                    sock.sendto(json.dumps({"command": "play_info", "duration": play_info["duration"]}).encode(), addr)
            elif command == "seek":
                manager.seek_stream(message.get("time"))
            elif command == "stop":
                manager.stop_stream()
            elif 'loss_rate' in message:
                manager.controller.update_strategy(message['loss_rate'])
        except Exception as e:
            print(f"[服务端-控制] 错误: {e}")

# --- RTP 简易打包 ---
def rtp_pack(seq, ts, payload):
    # 简化的RTP头: 2字节序列号, 4字节时间戳
    header = seq.to_bytes(2, 'big') + ts.to_bytes(4, 'big', signed=True)
    return header + payload

# --- 推流实现 (H.265重构) ---
def stream_from_camera(video_sock, audio_sock, controller, stream_control, client_addr):
    print("[服务端-推流] 摄像头推流激活 (H.265)。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[服务端-视频] 无法打开摄像头。")
        return

    start_time = time.time()

    # --- 音频线程 (基本无变化) ---
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE,
                                    input=True, frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[服务端-音频] 致命错误: 无法打开麦克风: {e}")
        p_audio.terminate()
        cap.release()
        return

    def audio_thread_func():
        audio_seq = 0
        while stream_control.get('running'):
            try:
                audio_data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                ts = int((time.time() - start_time) * 1000) # 时间戳 (毫秒)
                header = audio_seq.to_bytes(4, 'big') + ts.to_bytes(4, 'big', signed=True)
                audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                audio_seq = (audio_seq + 1) % (2**32)
            except IOError:
                break
        audio_stream.stop_stream()
        audio_stream.close()
        p_audio.terminate()
        print("[服务端-音频] 音频捕获线程已停止。")

    audio_thread = threading.Thread(target=audio_thread_func, daemon=True)
    audio_thread.start()

    # --- H.265 编码器设置 ---
    codec = av.codec.Codec(VIDEO_CODEC, 'w')
    encoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'w')
    video_seq = 0
    last_strategy = {}

    while stream_control.get('running'):
        ret, frame = cap.read()
        if not ret: break

        strategy = controller.get_current_strategy()

        # 当策略变化时，重新配置编码器
        if strategy != last_strategy:
            print(f"[服务端-编码器] 应用新策略: {strategy}")
            encoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'w')
            encoder.width = strategy['resolution'][0]
            encoder.height = strategy['resolution'][1]
            encoder.bit_rate = strategy['bitrate']
            encoder.framerate = strategy['fps_limit']
            encoder.pix_fmt = 'yuv420p' # H.265常用格式
            # 设置一些H.265编码选项以优化实时流
            encoder.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
            encoder.open(codec)
            last_strategy = strategy

        # 帧处理
        frame = cv2.resize(frame, strategy["resolution"])
        av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        av_frame.pts = int((time.time() - start_time) * 1000) # 使用毫秒级时间戳

        # 编码并发送
        try:
            for packet in encoder.encode(av_frame):
                # 简化的RTP打包
                rtp_packet = rtp_pack(video_seq, av_frame.pts, packet)
                video_sock.sendto(rtp_packet, (client_addr[0], VIDEO_PORT))
                video_seq = (video_seq + 1) % (2**16)
        except Exception as e:
            print(f"[服务端-编码器] 编码错误: {e}")

        time.sleep(1 / strategy["fps_limit"])

    cap.release()
    print("[服务端-推流] 摄像头推流结束。")


def stream_from_file(video_sock, audio_sock, controller, stream_control, client_addr, video_path):
    print(f"[服务端-推流] 文件推流激活 (H.265转码): {video_path}")
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"错误: 无法打开视频文件 {video_path} - {e}")
        return

    # --- H.265 编码器设置 ---
    codec = av.codec.Codec(VIDEO_CODEC, 'w')
    encoder = None # 在循环中根据策略动态创建
    last_strategy = {}

    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if not video_stream:
        print("文件中无视频流")
        container.close()
        return

    start_time = time.time()
    video_seq, audio_seq = 0, 0

    # 音频重采样器
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=AUDIO_RATE) if audio_stream else None
    audio_buffer = b''
    chunk_byte_size = AUDIO_CHUNK * 2 * AUDIO_CHANNELS # 16-bit = 2 bytes

    # 主循环
    while stream_control.get('running'):
        # 跳转逻辑
        if stream_control['seek_to'] >= 0:
            target_sec = stream_control['seek_to']
            stream_control['seek_to'] = -1.0
            try:
                # av.time_base is 1/1,000,000, so we seek in microseconds
                container.seek(int(target_sec * 1000000), backward=True, any_frame=False)
                print(f"[服务端-推流] 跳转成功到 {target_sec:.2f}s")
                start_time = time.time() - target_sec
                audio_buffer = b''

                # [修正] 关键修复：在跳转后，通过清空last_strategy来强制重新创建编码器。
                # 这可以清除编码器内部可能存在的旧状态，避免跳转后产生错误的PTS。
                last_strategy = {}
                encoder = None

            except Exception as e:
                print(f"[服务端-推流] 跳转失败: {e}")

        # 解码源文件包
        try:
            packet = next(container.demux(video_stream, audio_stream))
        except StopIteration:
            print("[服务端-推流] 文件播放结束。")
            break

        if packet.dts is None: continue

        current_pts_sec = float(packet.pts * packet.time_base)
        elapsed_time = time.time() - start_time

        # 同步播放时间
        if current_pts_sec > elapsed_time:
            time.sleep(max(0, current_pts_sec - elapsed_time))

        ts_ms = int(current_pts_sec * 1000)

        for frame in packet.decode():
            if not stream_control.get('running'): break

            # --- 视频处理 ---
            if isinstance(frame, av.VideoFrame):
                strategy = controller.get_current_strategy()

                # 动态调整编码器
                if strategy != last_strategy:
                    print(f"[服务端-编码器] 应用新策略: {strategy}")
                    encoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'w')
                    encoder.width = strategy['resolution'][0]
                    encoder.height = strategy['resolution'][1]
                    encoder.bit_rate = strategy['bitrate']
                    encoder.framerate = video_stream.average_rate
                    encoder.pix_fmt = 'yuv420p'
                    encoder.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
                    encoder.open(codec)
                    last_strategy = strategy

                if not encoder: continue

                # 改变分辨率并送入编码器
                resized_frame = frame.reformat(width=strategy['resolution'][0], height=strategy['resolution'][1], format='yuv420p')
                resized_frame.pts = ts_ms # 必须设置PTS

                for enc_packet in encoder.encode(resized_frame):
                    rtp_packet = rtp_pack(video_seq, ts_ms, enc_packet)
                    video_sock.sendto(rtp_packet, (client_addr[0], VIDEO_PORT))
                    video_seq = (video_seq + 1) % (2**16)

            # --- 音频处理 ---
            elif isinstance(frame, av.AudioFrame) and resampler:
                for resampled_frame in resampler.resample(frame):
                    audio_buffer += resampled_frame.to_ndarray().tobytes()

                while len(audio_buffer) >= chunk_byte_size:
                    chunk = audio_buffer[:chunk_byte_size]
                    audio_buffer = audio_buffer[chunk_byte_size:]
                    # 使用与视频帧相同的PTS计算方法，确保同步
                    audio_ts_ms = int((start_time + (time.time() - start_time)) * 1000)
                    header = audio_seq.to_bytes(4, 'big') + ts_ms.to_bytes(4, 'big', signed=True)
                    audio_sock.sendto(header + chunk, (client_addr[0], AUDIO_PORT))
                    audio_seq = (audio_seq + 1) % (2**32)

    container.close()
    print(f"[服务端-推流] 文件 {video_path} 推流结束。")


# --- 主函数 (无变化) ---
def main():
    sockets = {
        'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    }

    sockets['control'].bind((SERVER_HOST, CONTROL_PORT))
    controller = AdaptiveStreamController()
    manager = StreamerManager(sockets['video'], sockets['audio'], controller)

    control_thread = threading.Thread(target=control_channel_handler, args=(sockets['control'], manager), daemon=True)
    control_thread.start()

    print(f"[服务端] 服务器已启动于 {SERVER_HOST}:{CONTROL_PORT} (H.265/RTP模式)。按 Ctrl+C 关闭。")

    try:
        control_thread.join()
    except KeyboardInterrupt:
        print("\n[服务端] 关闭中...")
    finally:
        manager.stop_stream()
        [sock.close() for sock in sockets.values()]
        print("[服务端] 服务器已关闭。")

if __name__ == "__main__":
    main()