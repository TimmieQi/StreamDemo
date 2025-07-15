# server_av1_final.py
import cv2
import socket
import time
import threading
import json
import os
import av
import pyaudio
import struct

from shared_config import *


# --- 视频文件扫描 ---
def get_video_files(path="videos"):
    if not os.path.exists(path):
        os.makedirs(path)
        return []
    supported_formats = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    return [f for f in os.listdir(path) if f.endswith(supported_formats)]


# --- 自适应流控器 ---
class AdaptiveStreamController:
    def __init__(self):
        # AV1 可以在极低的比特率下达到可接受的画质
        self.configs = {
            "good": {"resolution": (640, 480), "bitrate": 600 * 1024, "fps_limit": 30},
            "medium": {"resolution": (640, 480), "bitrate": 300 * 1024, "fps_limit": 20},
            "poor": {"resolution": (480, 360), "bitrate": 150 * 1024, "fps_limit": 15}
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


# --- 推流管理器 (已更新) ---
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
                print("[服务端-管理器] 启动摄像头直播 (AV1)...")
                self.current_stream_thread = threading.Thread(
                    target=stream_av1_from_camera,
                    args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr),
                    daemon=True
                )
                self.current_stream_thread.start()
                return {"duration": 0}
            else:
                video_path = os.path.join("videos", source)
                if os.path.exists(video_path):
                    target_thread = None
                    try:
                        with av.open(video_path) as container:
                            # **重要修复**: 检查文件流的编码格式
                            video_stream = container.streams.video[0]
                            if video_stream.codec_context.name == 'av1':
                                print(f"[服务端-管理器] 文件 '{source}' 是AV1格式，将直接推流。")
                                target_thread = stream_av1_from_file
                            else:
                                print(
                                    f"[服务端-管理器] 文件 '{source}' 是 {video_stream.codec_context.name} 格式，将实时转码为AV1后推流。")
                                target_thread = transcode_to_av1_and_stream

                            duration_sec = container.duration / av.time_base
                    except Exception as e:
                        print(f"[服务端-管理器] 错误: 无法读取视频文件信息 {video_path}: {e}")
                        return None

                    self.current_stream_thread = threading.Thread(
                        target=target_thread,
                        args=(self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr,
                              video_path),
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


# --- AV1 推流实现 ---
def stream_av1_from_camera(video_sock, audio_sock, controller, stream_control, client_addr):
    # 此函数保持不变
    print("[服务端-推流] 摄像头AV1推流激活。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[服务端-视频] 无法打开摄像头。")
        return

    start_time = time.time()
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                                    frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[服务端-音频] 致命错误: 无法打开麦克风: {e}")
        p_audio.terminate()
        cap.release()
        return

    def audio_thread_func(stream_start_time):
        seq = 0
        while stream_control.get('running'):
            try:
                pts_ms = int((time.time() - stream_start_time) * 1000)
                audio_data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                header = seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
                audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                seq = (seq + 1) % (2 ** 64)
            except IOError:
                break
        audio_stream.close()
        p_audio.terminate()

    threading.Thread(target=audio_thread_func, args=(start_time,), daemon=True).start()

    strategy = controller.get_current_strategy()
    codec = av.codec.CodecContext.create('libaom-av1', 'w')
    codec.width = strategy['resolution'][0]
    codec.height = strategy['resolution'][1]
    codec.bit_rate = strategy['bitrate']
    codec.pix_fmt = 'yuv420p'
    codec.options = {'cpu-used': '8', 'deadline': 'realtime', 'row-mt': '1'}

    seq = 0
    while stream_control.get('running'):
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

        current_strategy = controller.get_current_strategy()
        if codec.width != current_strategy['resolution'][0] or codec.height != current_strategy['resolution'][1]:
            codec.width, codec.height = current_strategy['resolution'][0], current_strategy['resolution'][1]
        if codec.bit_rate != current_strategy['bitrate']:
            codec.bit_rate = current_strategy['bitrate']

        try:
            encoded_packets = codec.encode(av_frame)
        except Exception as e:
            print(f"[服务端-编码] 编码时发生错误: {e}")
            continue

        for packet in encoded_packets:
            pts_ms = int((time.time() - start_time) * 1000)
            header = seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
            video_sock.sendto(header + packet, (client_addr[0], VIDEO_PORT))
            seq = (seq + 1) % (2 ** 64)

        time.sleep(1 / current_strategy["fps_limit"])

    for packet in codec.encode(None):
        pts_ms = int((time.time() - start_time) * 1000)
        header = seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
        video_sock.sendto(header + packet, (client_addr[0], VIDEO_PORT))
        seq = (seq + 1) % (2 ** 64)

    cap.release()
    print("[服务端-推流] 摄像头AV1推流结束。")


def stream_av1_from_file(video_sock, audio_sock, controller, stream_control, client_addr, video_path):
    # 此函数现在只处理原生AV1文件
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"错误: 无法打开视频文件 {video_path} - {e}")
        return

    video_stream = next((s for s in container.streams.video if s.codec_context.name == 'av1'), None)
    audio_stream = next((s for s in container.streams.audio), None)

    if not video_stream:
        print(f"错误: {video_path} 中没有找到AV1视频流。")
        container.close()
        return

    print(f"[服务端-推流] 开始AV1文件直接推流: {video_path}")
    start_time = time.time()
    v_seq, a_seq = 0, 0
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono',
                                                  rate=AUDIO_RATE) if audio_stream else None

    while stream_control.get('running'):
        if stream_control['seek_to'] >= 0:
            target_sec = stream_control['seek_to']
            stream_control['seek_to'] = -1.0
            try:
                container.seek(int(target_sec * av.time_base))
                print(f"[服务端-推流] 跳转成功到 {target_sec:.2f}s")
                start_time = time.time() - target_sec
            except Exception as e:
                print(f"[服务端-推流] 跳转失败: {e}")

        try:
            packet = next(container.demux(video_stream, audio_stream))
        except StopIteration:
            print("[服务端-推流] 文件播放结束。")
            break

        if packet.dts is None: continue
        current_pts_sec = float(packet.pts * packet.time_base)
        elapsed_time = time.time() - start_time
        if current_pts_sec > elapsed_time:
            time.sleep(max(0, current_pts_sec - elapsed_time))
        pts_ms = int(current_pts_sec * 1000)

        if not stream_control.get('running'): break

        if packet.stream.type == 'video':
            header = v_seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
            video_sock.sendto(header + packet, (client_addr[0], VIDEO_PORT))
            v_seq = (v_seq + 1) % (2 ** 64)

        elif packet.stream.type == 'audio' and resampler:
            for frame in packet.decode():
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    audio_data = resampled_frame.to_ndarray().tobytes()
                    header = a_seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
                    audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                    a_seq = (a_seq + 1) % (2 ** 64)

    container.close()
    print(f"[服务端-推流] 文件 {video_path} AV1直接推流结束。")


def transcode_to_av1_and_stream(video_sock, audio_sock, controller, stream_control, client_addr, video_path):
    # **全新函数**: 用于将非AV1文件转码为AV1并推流
    try:
        input_container = av.open(video_path)
    except Exception as e:
        print(f"错误: 无法打开输入文件 {video_path} - {e}")
        return

    input_video_stream = input_container.streams.video[0]
    input_audio_stream = input_container.streams.audio[0] if input_container.streams.audio else None

    print(f"[服务端-转码] 开始将 {video_path} 转码为AV1并推流...")

    # 设置AV1编码器
    strategy = controller.get_current_strategy()
    output_codec = av.codec.CodecContext.create('libaom-av1', 'w')
    output_codec.width = strategy['resolution'][0]
    output_codec.height = strategy['resolution'][1]
    output_codec.bit_rate = strategy['bitrate']
    output_codec.pix_fmt = 'yuv420p'
    output_codec.options = {'cpu-used': '8', 'deadline': 'realtime', 'row-mt': '1'}

    start_time = time.time()
    v_seq, a_seq = 0, 0
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono',
                                                  rate=AUDIO_RATE) if input_audio_stream else None

    while stream_control.get('running'):
        if stream_control['seek_to'] >= 0:
            # 实时转码时，精确跳转很复杂，此处简化为不支持
            print("[服务端-转码] 实时转码模式下不支持跳转。")
            stream_control['seek_to'] = -1.0

        try:
            packet = next(input_container.demux(input_video_stream, input_audio_stream))
        except StopIteration:
            print("[服务端-转码] 文件转码播放结束。")
            break

        if packet.dts is None: continue
        current_pts_sec = float(packet.pts * packet.time_base)
        elapsed_time = time.time() - start_time
        if current_pts_sec > elapsed_time:
            time.sleep(max(0, current_pts_sec - elapsed_time))
        pts_ms = int(current_pts_sec * 1000)

        if not stream_control.get('running'): break

        if packet.stream.type == 'video':
            for frame in packet.decode():
                # 调整分辨率
                resized_frame = frame.reformat(width=strategy['resolution'][0], height=strategy['resolution'][1],
                                               format='yuv420p')
                for out_packet in output_codec.encode(resized_frame):
                    header = v_seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
                    video_sock.sendto(header + out_packet, (client_addr[0], VIDEO_PORT))
                    v_seq = (v_seq + 1) % (2 ** 64)

        elif packet.stream.type == 'audio' and resampler:
            for frame in packet.decode():
                resampled_frames = resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    audio_data = resampled_frame.to_ndarray().tobytes()
                    header = a_seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
                    audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                    a_seq = (a_seq + 1) % (2 ** 64)

    # 清空编码器缓冲区
    for out_packet in output_codec.encode(None):
        header = v_seq.to_bytes(8, 'big') + pts_ms.to_bytes(8, 'big', signed=True)
        video_sock.sendto(header + out_packet, (client_addr[0], VIDEO_PORT))
        v_seq = (v_seq + 1) % (2 ** 64)

    input_container.close()
    print(f"[服务端-转码] 文件 {video_path} 转码推流结束。")


# --- 主函数与控制信道 ---
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


def main():
    sockets = {
        'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    }

    sockets['control'].bind((SERVER_HOST, CONTROL_PORT))
    controller = AdaptiveStreamController()
    manager = StreamerManager(sockets['video'], sockets['audio'], controller)

    control_thread = threading.Thread(
        target=control_channel_handler,
        args=(sockets['control'], manager),
        daemon=True
    )
    control_thread.start()

    print(f"[服务端] AV1服务器已启动于 {SERVER_HOST}:{CONTROL_PORT}。按 Ctrl+C 关闭。")

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
