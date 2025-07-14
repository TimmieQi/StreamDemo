# server.py

import io
import cv2
import socket
import time
import threading
import json
import os
import av
import pyaudio
import numpy as np # 需要 numpy 来进行数据切片

from shared_config import *


# --- 视频文件扫描 (无变化) ---
def get_video_files(path="videos"):
    if not os.path.exists(path):
        print(f"警告: 视频目录 '{path}' 不存在。将创建一个空目录。")
        os.makedirs(path)
        return []
    supported_formats = ('.mp4', '.mkv', '.avi', '.mov')
    return [f for f in os.listdir(path) if f.endswith(supported_formats)]


# --- 自适应流控器 (无变化) ---
class AdaptiveStreamController:
    def __init__(self):
        self.configs = {
            "good": {"resolution": (640, 480), "jpeg_quality": 85, "fps_limit": 30},
            "medium": {"resolution": (480, 360), "jpeg_quality": 60, "fps_limit": 20},
            "poor": {"resolution": (320, 240), "jpeg_quality": 40, "fps_limit": 15},
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
        self.running_flag = None
        self.lock = threading.Lock()

    def start_stream(self, source, client_addr):
        with self.lock:
            print("[服务端-管理器] 请求开启新推流...")
            self.stop_stream()
            self.running_flag = {'running': True}
            if source == "camera":
                print("[服务端-管理器] 启动摄像头直播...")
                self.current_stream_thread = threading.Thread(
                    target=stream_from_camera,
                    args=(self.video_sock, self.audio_sock, self.controller, self.running_flag, client_addr),
                    daemon=True
                )
            else:
                video_path = os.path.join("videos", source)
                if os.path.exists(video_path):
                    print(f"[服务端-管理器] 启动文件点播: {source}")
                    self.current_stream_thread = threading.Thread(
                        target=stream_from_file,
                        args=(
                            self.video_sock, self.audio_sock, self.controller, self.running_flag, client_addr,
                            video_path),
                        daemon=True
                    )
                else:
                    print(f"[服务端-管理器] 错误: 找不到视频文件 {video_path}")
                    return
            self.current_stream_thread.start()

    def stop_stream(self):
        if self.running_flag:
            self.running_flag['running'] = False
            print("[服务端-管理器] 发送停止信号到当前推流线程...")
        if self.current_stream_thread and self.current_stream_thread.is_alive():
            print("[服务端-管理器] 等待推流线程结束...")
            self.current_stream_thread.join(timeout=1.0)
        self.current_stream_thread = None
        self.running_flag = None
        print("[服务端-管理器] 推流已确认停止。")


# --- 控制信道处理 (无变化) ---
def control_channel_handler(sock, manager):
    client_info = {'addr': None, 'last_contact': 0}

    def watchdog():
        while True:
            time.sleep(5)
            if client_info['addr'] and (time.time() - client_info['last_contact'] > 5):
                print(f"[服务端-看门狗] 客户端 {client_info['addr']} 超时，停止推流。")
                manager.stop_stream()
                client_info['addr'] = None

    threading.Thread(target=watchdog, daemon=True).start()

    video_list = get_video_files()
    print(f"[服务端-控制] 可用视频文件: {video_list}")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            client_info.update({'addr': addr, 'last_contact': time.time()})
            message = json.loads(data.decode())
            command = message.get("command")

            if command == "get_list":
                sock.sendto(json.dumps(["camera"] + video_list).encode(), addr)
            elif command == "play":
                manager.start_stream(message.get("source"), addr)
            elif command == "stop":
                manager.stop_stream()
            elif 'loss_rate' in message:
                manager.controller.update_strategy(message['loss_rate'])
        except Exception as e:
            print(f"[服务端-控制] 错误: {e}")


# --- 推流实现 (修复文件推流的音频分包逻辑) ---
def stream_from_camera(video_sock, audio_sock, controller, running_flag, client_addr):
    print("[服务端-推流] 摄像头推流激活。")
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                                    rate=AUDIO_RATE, input=True,
                                    frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[服务端-音频] 致命错误: 无法打开麦克风: {e}")
        p_audio.terminate()
        return

    def audio_thread_func():
        print("[服务端-音频] 音频捕获线程已启动。")
        sequence_number = 0
        while running_flag.get('running'):
            try:
                audio_data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                header = sequence_number.to_bytes(8, 'big')
                audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                sequence_number = (sequence_number + 1) % (2**64)
            except IOError as e:
                print(f"[服务端-音频] 音频捕获IO错误: {e}")
                break
        print("[服务端-音频] 音频捕获线程停止。")

    threading.Thread(target=audio_thread_func, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[服务端-视频] 无法打开摄像头。")
        running_flag['running'] = False

    frame_id = 0
    while running_flag.get('running') and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        strategy = controller.get_current_strategy()
        frame = cv2.resize(frame, strategy["resolution"])
        _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), strategy["jpeg_quality"]])
        frame_data = encoded_frame.tobytes()

        total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
        for i in range(total_packets):
            chunk = frame_data[i * PACKET_SIZE: (i + 1) * PACKET_SIZE]
            header = frame_id.to_bytes(4, 'big') + i.to_bytes(2, 'big') + total_packets.to_bytes(2, 'big')
            video_sock.sendto(header + chunk, (client_addr[0], VIDEO_PORT))

        frame_id = (frame_id + 1) % (2 ** 32 - 1)
        time.sleep(1 / strategy["fps_limit"])

    running_flag['running'] = False
    if audio_stream.is_active(): audio_stream.stop_stream()
    audio_stream.close()
    p_audio.terminate()
    if cap.isOpened(): cap.release()
    print("[服务端-推流] 摄像头推流结束。")


def stream_from_file(video_sock, audio_sock, controller, running_flag, client_addr, video_path):
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"错误: 无法打开视频文件 {video_path} - {e}")
        return

    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    if not video_stream:
        print(f"错误: {video_path} 中没有视频流。")
        container.close()
        return

    print(f"[服务端-推流] 开始推流文件: {video_path}")
    start_time = time.time()
    frame_id = 0
    audio_sequence_number = 0

    streams_to_demux = [s for s in [video_stream, audio_stream] if s]

    resampler = None
    if audio_stream:
        print(f"[服务端-音频] 文件包含音频流。将重采样至 {AUDIO_RATE}Hz, 16-bit Mono。")
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=AUDIO_RATE)
    else:
        print(f"[服务端-音频] [警告] {video_path} 中没有音频流。")

    # ### 核心修复：引入一个缓冲区来存储未发送完的音频数据 ###
    audio_buffer = b''
    # 每个样本是16位(2字节)，单声道
    chunk_byte_size = AUDIO_CHUNK * 2 * AUDIO_CHANNELS

    for packet in container.demux(streams_to_demux):
        if not running_flag.get('running'): break
        if packet.dts is None: continue

        current_pts = float(packet.pts * packet.time_base)
        elapsed_time = time.time() - start_time
        if current_pts > elapsed_time:
            time.sleep(max(0, current_pts - elapsed_time))

        for frame in packet.decode():
            if not running_flag.get('running'): break
            if packet.stream.type == 'video':
                img = frame.to_image()
                strategy = controller.get_current_strategy()
                img = img.resize(strategy["resolution"])
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=strategy["jpeg_quality"])
                frame_data = buffer.getvalue()

                total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
                for i in range(total_packets):
                    chunk = frame_data[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
                    header = frame_id.to_bytes(4, 'big') + i.to_bytes(2, 'big') + total_packets.to_bytes(2, 'big')
                    video_sock.sendto(header + chunk, (client_addr[0], VIDEO_PORT))
                frame_id = (frame_id + 1) % (2 ** 32 - 1)

            elif packet.stream.type == 'audio' and resampler:
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # 将新解码的数据加入缓冲区
                    audio_buffer += resampled_frame.to_ndarray().tobytes()

                    # ### 核心修复：循环切分缓冲区，发送固定大小的块 ###
                    while len(audio_buffer) >= chunk_byte_size:
                        # 从缓冲区头部取出一个标准大小的块
                        audio_chunk_to_send = audio_buffer[:chunk_byte_size]
                        # 更新缓冲区，移除已取出的部分
                        audio_buffer = audio_buffer[chunk_byte_size:]

                        # 发送这个标准大小的块
                        header = audio_sequence_number.to_bytes(8, 'big')
                        audio_sock.sendto(header + audio_chunk_to_send, (client_addr[0], AUDIO_PORT))
                        audio_sequence_number = (audio_sequence_number + 1) % (2**64)

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

    print(f"[服务端] 服务器已启动于 {SERVER_HOST}:{CONTROL_PORT}。按 Ctrl+C 关闭。")
    try:
        control_thread.join()
    except KeyboardInterrupt:
        print("\n[服务端] 关闭中...")
    finally:
        manager.stop_stream()
        for sock in sockets.values():
            sock.close()
        print("[服务端] 服务器已关闭。")


if __name__ == "__main__":
    main()