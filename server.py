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
import numpy as np

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
        with self.lock: return self.configs[self.current_strategy_name]

    def update_strategy(self, loss_rate):
        new_strategy_name = "good"
        if loss_rate >= 0.1: new_strategy_name = "poor"
        elif loss_rate >= 0.03: new_strategy_name = "medium"
        with self.lock:
            if self.current_strategy_name != new_strategy_name:
                print(f"[服务端-控制器] 丢包率: {loss_rate:.2%}, 切换策略至: {new_strategy_name.upper()}")
                self.current_strategy_name = new_strategy_name


# --- 推流管理器 (修改以支持seek) ---
class StreamerManager:
    def __init__(self, video_sock, audio_sock, controller):
        self.video_sock, self.audio_sock = video_sock, audio_sock
        self.controller = controller
        self.current_stream_thread = None
        # ### 修改：使用一个控制字典来与推流线程通信 ###
        self.stream_control = {'running': False, 'seek_to': -1.0}
        self.lock = threading.Lock()

    def start_stream(self, source, client_addr):
        with self.lock:
            print("[服务端-管理器] 请求开启新推流...")
            self.stop_stream() # 停止旧的
            self.stream_control = {'running': True, 'seek_to': -1.0} # 重置控制状态

            if source == "camera":
                print("[服务端-管理器] 启动摄像头直播...")
                # 注意：摄像头直播不支持seek
                self.current_stream_thread = threading.Thread(
                    target=stream_from_camera,
                    args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr),
                    daemon=True)
                self.current_stream_thread.start()
                return {"duration": 0} # 摄像头直播时长为0
            else:
                video_path = os.path.join("videos", source)
                if os.path.exists(video_path):
                    print(f"[服务端-管理器] 启动文件点播: {source}")
                    try:
                        # 预先打开文件以获取时长
                        with av.open(video_path) as container:
                            duration_sec = container.duration / av.time_base
                    except Exception as e:
                        print(f"[服务端-管理器] 错误: 无法读取视频文件信息 {video_path}: {e}")
                        return None

                    self.current_stream_thread = threading.Thread(
                        target=stream_from_file,
                        args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr, video_path),
                        daemon=True)
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


# --- 控制信道处理 (修改以支持新指令) ---
def control_channel_handler(sock, manager):
    client_info = {'addr': None, 'last_contact': 0}
    video_files = get_video_files()

    def watchdog():
        while True:
            time.sleep(5)
            if client_info['addr'] and (time.time() - client_info['last_contact'] > 5):
                print(f"[服务端-看门狗] 客户端 {client_info['addr']} 超时，停止推流。")
                manager.stop_stream(); client_info['addr'] = None

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
                    response = {"command": "play_info", "duration": play_info["duration"]}
                    sock.sendto(json.dumps(response).encode(), addr)
            elif command == "seek":
                manager.seek_stream(message.get("time"))
            elif command == "stop":
                manager.stop_stream()
            elif 'loss_rate' in message:
                manager.controller.update_strategy(message['loss_rate'])
        except Exception as e:
            print(f"[服务端-控制] 错误: {e}")


# --- 推流实现 (修改以支持seek) ---
def stream_from_camera(video_sock, audio_sock, controller, stream_control, client_addr):
    # 此函数基本不变，只修改了循环条件
    start_time = time.time()
    p_audio = pyaudio.PyAudio()
    audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK)

    def audio_thread_func(stream_start_time):
        sequence_number = 0
        while stream_control.get('running'):
            pts_ms = int((time.time() - stream_start_time) * 1000)
            audio_data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            header = sequence_number.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big')
            audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
            sequence_number = (sequence_number + 1) % (2**64)

    threading.Thread(target=audio_thread_func, args=(start_time,), daemon=True).start()
    cap = cv2.VideoCapture(0)
    frame_id = 0
    while stream_control.get('running') and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # ... (视频发送逻辑不变) ...
    # ... (清理逻辑不变) ...


def stream_from_file(video_sock, audio_sock, controller, stream_control, client_addr, video_path):
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"错误: 无法打开视频文件 {video_path} - {e}"); return

    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not video_stream: print(f"错误: {video_path} 中没有视频流。"); container.close(); return

    print(f"[服务端-推流] 开始推流文件: {video_path}")
    start_time = time.time()
    frame_id, audio_sequence_number = 0, 0
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=AUDIO_RATE) if audio_stream else None
    audio_buffer = b''; chunk_byte_size = AUDIO_CHUNK * 2 * AUDIO_CHANNELS

    # 将demuxer放入循环，以便seek后可以重新迭代
    while stream_control.get('running'):
        # ### seek逻辑 ###
        if stream_control['seek_to'] >= 0:
            target_sec = stream_control['seek_to']
            stream_control['seek_to'] = -1.0 # 重置seek请求
            try:
                # av.Container.seek() 需要时间戳和流。时间戳单位是stream.time_base
                target_pts = int(target_sec * av.time_base)
                container.seek(target_pts)
                print(f"[服务端-推流] 跳转成功到 {target_sec:.2f}s")
                # 重置时间同步逻辑
                start_time = time.time() - target_sec
                # 清空音频缓冲区，因为里面的数据是旧的
                audio_buffer = b''
            except Exception as e:
                print(f"[服务端-推流] 跳转失败: {e}")

        # 从容器中解包
        try:
            packet = next(container.demux(video_stream, audio_stream))
        except StopIteration:
            print("[服务端-推流] 文件播放结束。")
            break # 退出外层while循环

        if packet.dts is None: continue

        current_pts_sec = float(packet.pts * packet.time_base)
        elapsed_time = time.time() - start_time
        if current_pts_sec > elapsed_time:
            time.sleep(max(0, current_pts_sec - elapsed_time))

        pts_ms = int(current_pts_sec * 1000)

        for frame in packet.decode():
            if not stream_control.get('running'): break
            if packet.stream.type == 'video':
                # ... (视频打包发送逻辑不变, 仍然发送pts_ms) ...
                img = frame.to_image()
                strategy = controller.get_current_strategy()
                img = img.resize(strategy["resolution"])
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=strategy["jpeg_quality"])
                frame_data = buffer.getvalue()

                total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
                for i in range(total_packets):
                    chunk = frame_data[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
                    header = frame_id.to_bytes(4, 'big') + pts_ms.to_bytes(8, 'big') + \
                             i.to_bytes(2, 'big') + total_packets.to_bytes(2, 'big')
                    video_sock.sendto(header + chunk, (client_addr[0], VIDEO_PORT))
                frame_id = (frame_id + 1) % (2 ** 32 - 1)

            elif packet.stream.type == 'audio' and resampler:
                # ... (音频分块发送逻辑不变, 仍然发送pts_ms) ...
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    audio_buffer += resampled_frame.to_ndarray().tobytes()
                    while len(audio_buffer) >= chunk_byte_size:
                        audio_chunk_to_send = audio_buffer[:chunk_byte_size]
                        audio_buffer = audio_buffer[chunk_byte_size:]
                        header = audio_sequence_number.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big')
                        audio_sock.sendto(header + audio_chunk_to_send, (client_addr[0], AUDIO_PORT))
                        audio_sequence_number = (audio_sequence_number + 1) % (2**64)

    container.close()
    print(f"[服务端-推流] 文件 {video_path} 推流结束。")


# --- 主函数 (无变化) ---
def main():
    sockets = {'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)}
    sockets['control'].bind((SERVER_HOST, CONTROL_PORT))
    controller = AdaptiveStreamController()
    manager = StreamerManager(sockets['video'], sockets['audio'], controller)
    control_thread = threading.Thread(target=control_channel_handler, args=(sockets['control'], manager), daemon=True)
    control_thread.start()
    print(f"[服务端] 服务器已启动于 {SERVER_HOST}:{CONTROL_PORT}。按 Ctrl+C 关闭。")
    try: control_thread.join()
    except KeyboardInterrupt: print("\n[服务端] 关闭中...")
    finally:
        manager.stop_stream()
        for sock in sockets.values(): sock.close()
        print("[服务端] 服务器已关闭。")

if __name__ == "__main__":
    main()