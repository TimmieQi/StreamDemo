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

# [NEW] 定义网络包的最大尺寸，以字节为单位。1400是一个安全值，避免IP分片。
PACKET_MAX_SIZE = 1400


# --- 视频文件扫描 (无变化) ---
def get_video_files(path="videos"):
    if not os.path.exists(path):
        os.makedirs(path)
        return []
    supported_formats = ('.mp4', '.mkv', '.avi', '.mov')
    return [f for f in os.listdir(path) if f.endswith(supported_formats)]


# --- 自适应流控器 (无变化) ---
class AdaptiveStreamController:
    def __init__(self):
        self.configs = {
            "good": {"multiplier": 1.0, "fps_limit": 60},
            "medium": {"multiplier": 0.5, "fps_limit": 30},
            "poor": {"multiplier": 0.25, "fps_limit": 20}
        }
        self.current_strategy_name = "good"
        self.lock = threading.Lock()

    def get_current_strategy(self):
        with self.lock: return self.configs[self.current_strategy_name]

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
        self.video_sock = video_sock;
        self.audio_sock = audio_sock;
        self.controller = controller
        self.current_stream_thread = None;
        self.stream_control = {'running': False, 'seek_to': -1.0}
        self.lock = threading.Lock()

    def start_stream(self, source, client_addr):
        with self.lock:
            print("[服务端-管理器] 请求开启新推流...")
            self.stop_stream()
            self.stream_control = {'running': True, 'seek_to': -1.0}
            if source == "camera":
                print("[服务端-管理器] 启动摄像头直播...")
                self.current_stream_thread = threading.Thread(target=stream_from_camera, args=(
                self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr), daemon=True)
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
                        print(f"[服务端-管理器] 错误: 无法读取视频文件信息 {video_path}: {e}"); return None
                    self.current_stream_thread = threading.Thread(target=stream_from_file, args=(
                    self.video_sock, self.audio_sock, self.controller, self.stream_control, client_addr, video_path),
                                                                  daemon=True)
                    self.current_stream_thread.start()
                    return {"duration": duration_sec}
                else:
                    print(f"[服务端-管理器] 错误: 找不到视频文件 {video_path}"); return None

    def stop_stream(self):
        if self.stream_control['running']: self.stream_control['running'] = False; print(
            "[服务端-管理器] 发送停止信号到当前推流线程...")
        if self.current_stream_thread and self.current_stream_thread.is_alive(): self.current_stream_thread.join(
            timeout=1.0)
        self.current_stream_thread = None;
        print("[服务端-管理器] 推流已确认停止。")

    def seek_stream(self, target_time_sec):
        with self.lock:
            if self.stream_control['running']: print(f"[服务端-管理器] 请求跳转到 {target_time_sec:.2f} 秒");
            self.stream_control['seek_to'] = target_time_sec


# --- 控制信道处理 (无变化) ---
def control_channel_handler(sock, manager):
    client_info = {'addr': None, 'last_contact': 0};
    video_files = get_video_files()

    def watchdog():
        while True:
            time.sleep(5)
            if client_info['addr'] and (time.time() - client_info['last_contact'] > 5):
                print(f"[服务端-看门狗] 客户端 {client_info['addr']} 超时，停止推流。");
                manager.stop_stream();
                client_info['addr'] = None

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[服务端-控制] 可用视频文件: {video_files}")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            client_info.update({'addr': addr, 'last_contact': time.time()})
            message = json.loads(data.decode());
            command = message.get("command")
            if command == "get_list":
                sock.sendto(json.dumps(["camera"] + video_files).encode(), addr)
            elif command == "play":
                play_info = manager.start_stream(message.get("source"), addr)
                if play_info: sock.sendto(
                    json.dumps({"command": "play_info", "duration": play_info["duration"]}).encode(), addr)
            elif command == "seek":
                manager.seek_stream(message.get("time"))
            elif command == "stop":
                manager.stop_stream()
            elif 'loss_rate' in message:
                manager.controller.update_strategy(message['loss_rate'])
        except Exception as e:
            print(f"[服务端-控制] 错误: {e}")



def send_packet_fragmented(sock, client_addr, seq, ts, packet):
    """
    发送可能需要分片的RTP包。
    新版协议: Header (7 bytes) = Seq (2B) + TS (4B) + FragInfo (1B)
    FragInfo:
     - 0x80 (1000 0000): Start of Frame
     - 0x40 (0100 0000): End of Frame
    """
    header_size = 7  # 2B seq + 4B ts + 1B frag_info
    max_payload_size = PACKET_MAX_SIZE - header_size

    packet_bytes = bytes(packet)

    if len(packet_bytes) <= max_payload_size:
        # 包不大，作为单个完整包发送 (S=1, E=1)
        frag_info = 0b11000000  # S=1, E=1
        header = (
                seq.to_bytes(2, 'big') +
                ts.to_bytes(4, 'big', signed=True) +
                frag_info.to_bytes(1, 'big')
        )
        sock.sendto(header + packet_bytes, client_addr)
        return 1  # 返回发送的包数量

    # 包太大，需要分片
    num_packets = 0
    offset = 0
    while offset < len(packet_bytes):
        chunk = packet_bytes[offset:offset + max_payload_size]

        frag_info = 0
        is_first = (offset == 0)
        is_last = (offset + len(chunk) >= len(packet_bytes))

        if is_first:
            frag_info |= 0b10000000  # Set Start bit
        if is_last:
            frag_info |= 0b01000000  # Set End bit

        current_seq = (seq + num_packets) % (2 ** 16)
        header = (
                current_seq.to_bytes(2, 'big') +
                ts.to_bytes(4, 'big', signed=True) +
                frag_info.to_bytes(1, 'big')
        )
        sock.sendto(header + chunk, client_addr)

        offset += len(chunk)
        num_packets += 1

    # print(f"Frame (TS:{ts}) fragmented into {num_packets} packets.")
    return num_packets


# --- 推流实现 (已修改以使用分包逻辑) ---
def stream_from_camera(video_sock, audio_sock, controller, stream_control, client_addr):
    print("[服务端-推流] 摄像头推流激活 (H.265, 动态码率)。")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("[服务端-视频] 无法打开摄像头。"); return

    ret, frame = cap.read()
    if not ret: print("[服务端-视频] 无法从摄像头读取帧。"); cap.release(); return
    source_resolution = (frame.shape[1], frame.shape[0])
    BASE_BITRATE = 1500 * 1024
    print(f"[服务端-视频] 摄像头源分辨率: {source_resolution}, 基础码率: {BASE_BITRATE / 1024} kbps")
    start_time = time.time()

    # ... 音频线程代码无变化，此处省略 ...
    p_audio = pyaudio.PyAudio()
    try:
        audio_stream = p_audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                                    frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[服务端-音频] 致命错误: 无法打开麦克风: {e}"); p_audio.terminate(); cap.release(); return

    def audio_thread_func():
        audio_seq = 0
        while stream_control.get('running'):
            try:
                audio_data = audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                ts = int((time.time() - start_time) * 1000)
                header = audio_seq.to_bytes(4, 'big') + ts.to_bytes(4, 'big', signed=True)
                audio_sock.sendto(header + audio_data, (client_addr[0], AUDIO_PORT))
                audio_seq = (audio_seq + 1) % (2 ** 32)
            except IOError:
                break
        audio_stream.stop_stream();
        audio_stream.close();
        p_audio.terminate();
        print("[服务端-音频] 音频捕获线程已停止。")

    audio_thread = threading.Thread(target=audio_thread_func, daemon=True);
    audio_thread.start()

    codec = av.codec.Codec(VIDEO_CODEC, 'w')
    encoder, video_seq, last_strategy = None, 0, {}

    while stream_control.get('running'):
        if not ret: ret, frame = cap.read()
        if not ret: break

        strategy = controller.get_current_strategy()
        if strategy != last_strategy:
            print(f"[服务端-编码器] 应用新策略: {strategy}")
            encoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'w')
            encoder.width, encoder.height = source_resolution
            encoder.bit_rate = int(BASE_BITRATE * strategy['multiplier'])
            encoder.framerate = strategy['fps_limit']
            encoder.pix_fmt = 'yuv420p'
            encoder.options = {'preset': 'ultrafast',
                               'tune': 'zerolatency',
                               'bframes': '0'}
            encoder.open(codec);
            last_strategy = strategy

        if not encoder: continue

        av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        av_frame.pts = int((time.time() - start_time) * 1000)

        try:
            for packet in encoder.encode(av_frame):
                # [MODIFIED] 使用分包函数发送
                num_sent = send_packet_fragmented(video_sock, (client_addr[0], VIDEO_PORT), video_seq, av_frame.pts,
                                                  packet)
                video_seq = (video_seq + num_sent) % (2 ** 16)
        except Exception as e:
            print(f"[服务端-编码器] 编码错误: {e}")

        ret = False
        time.sleep(1 / strategy["fps_limit"])

    cap.release()
    print("[服务端-推流] 摄像头推流结束。")


def stream_from_file(video_sock, audio_sock, controller, stream_control, client_addr, video_path):
    print(f"[服务端-推流] 文件推流激活 (H.265转码, 动态码率): {video_path}")
    try:
        container = av.open(video_path, 'r')
    except Exception as e:
        print(f"错误: 无法打开视频文件 {video_path} - {e}"); return

    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not video_stream:
        print("文件中无视频流"); container.close(); return

    # [NEW] 解码后帧的排序缓冲区，解决B帧导致的解码顺序与显示顺序不一致问题
    # 格式: [(pts, frame_object), ...]
    decoded_frame_buffer = []

    source_resolution = (video_stream.width, video_stream.height)
    BASE_BITRATE = 1500 * 1024
    print(f"[服务端-视频] 文件源分辨率: {source_resolution}, 基础码率: {BASE_BITRATE / 1024} kbps")

    codec = av.codec.Codec(VIDEO_CODEC, 'w')
    encoder, last_strategy = None, {}
    video_seq, audio_seq = 0, 0

    resampler = av.audio.resampler.AudioResampler(
        format='s16', layout='mono', rate=AUDIO_RATE
    ) if audio_stream else None
    audio_buffer = b''
    chunk_byte_size = AUDIO_CHUNK * 2 * AUDIO_CHANNELS

    start_time_perf = time.perf_counter()
    start_pts_sec = 0.0
    first_frame_processed = False

    demuxer = container.demux(video_stream, audio_stream)

    while stream_control.get('running'):
        # --- Seek Logic ---
        if stream_control['seek_to'] >= 0:
            target_sec = stream_control['seek_to']
            stream_control['seek_to'] = -1.0
            try:
                container.seek(int(target_sec / video_stream.time_base), backward=True, stream=video_stream)
                print(f"[服务端-推流] 跳转到 ~{target_sec:.2f}s")
                # 清空所有缓冲区和状态
                decoded_frame_buffer.clear()
                audio_buffer = b''
                last_strategy = {}
                encoder = None
                first_frame_processed = False
                demuxer = container.demux(video_stream, audio_stream) # 重置demuxer
            except (StopIteration, av.AVError) as e:
                print(f"[服务端-推流] 跳转失败: {e}")

        # --- Frame Buffering Logic ---
        # 当缓冲区中的帧数较少时，持续解码并填充缓冲区
        # 缓冲一定数量的帧（例如，1秒的量）以确保有足够的帧进行排序
        while len(decoded_frame_buffer) < video_stream.average_rate * 1 and stream_control.get('running'):
            try:
                packet = next(demuxer)
                if packet.dts is None: continue

                # 解码packet得到一个或多个frame
                for frame in packet.decode():
                    current_pts_sec = float(frame.pts * frame.time_base)
                    # 将解码后的帧（视频或音频）连同其PTS存入缓冲区
                    decoded_frame_buffer.append((current_pts_sec, frame))

            except StopIteration:
                # 文件读取完毕，跳出填充循环
                break

        # 如果填充后缓冲区依然为空（意味着文件已播放完毕），则结束推流
        if not decoded_frame_buffer:
            print("[服务端-推流] 文件播放结束。")
            break

        # --- Frame Processing Logic ---
        # 对缓冲区中的所有帧按PTS（显示时间戳）进行排序
        decoded_frame_buffer.sort(key=lambda x: x[0])

        # 从缓冲区头部取出一个帧进行处理
        current_pts_sec, frame = decoded_frame_buffer.pop(0)

        # 设置计时器起点
        if not first_frame_processed:
            start_pts_sec = current_pts_sec
            start_time_perf = time.perf_counter()
            first_frame_processed = True

        # 精确计时和发送
        target_elapsed_time = current_pts_sec - start_pts_sec
        real_elapsed_time = time.perf_counter() - start_time_perf
        wait_time = target_elapsed_time - real_elapsed_time
        if wait_time > 0:
            time.sleep(wait_time)

        ts_ms = int(current_pts_sec * 1000)

        # --- Process Video Frame ---
        if isinstance(frame, av.VideoFrame):
            strategy = controller.get_current_strategy()
            if strategy != last_strategy:
                # ... (编码器创建逻辑不变) ...
                print(f"[服务端-编码器] 应用新策略: {strategy}")
                encoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'w')
                encoder.width, encoder.height = source_resolution
                encoder.bit_rate = int(BASE_BITRATE * strategy['multiplier'])
                encoder.framerate = video_stream.average_rate
                encoder.pix_fmt = 'yuv420p'
                encoder.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'bframes': '0'}
                encoder.open(codec)
                last_strategy = strategy

            if not encoder: continue

            # 高效、可靠地“净化”帧
            ndarray_yuv = frame.to_ndarray(format='yuv420p')
            clean_frame = av.VideoFrame.from_ndarray(ndarray_yuv, format='yuv420p')
            clean_frame.pts = ts_ms

            for enc_packet in encoder.encode(clean_frame):
                num_sent = send_packet_fragmented(video_sock, (client_addr[0], VIDEO_PORT), video_seq, ts_ms, enc_packet)
                video_seq = (video_seq + num_sent) % (2 ** 16)

        # --- Process Audio Frame ---
        elif isinstance(frame, av.AudioFrame) and resampler:
            for resampled_frame in resampler.resample(frame):
                audio_buffer += resampled_frame.to_ndarray().tobytes()
            while len(audio_buffer) >= chunk_byte_size:
                chunk = audio_buffer[:chunk_byte_size]
                audio_buffer = audio_buffer[chunk_byte_size:]
                header = audio_seq.to_bytes(4, 'big') + ts_ms.to_bytes(4, 'big', signed=True)
                audio_sock.sendto(header + chunk, (client_addr[0], AUDIO_PORT))
                audio_seq = (audio_seq + 1) % (2 ** 32)

    # 冲洗编码器，发送剩余的帧
    if encoder:
        for packet in encoder.flush():
            # 这里需要一个ts，可以用最后一个已知ts
            last_ts_ms = int(current_pts_sec * 1000) if 'current_pts_sec' in locals() else 0
            num_sent = send_packet_fragmented(video_sock, (client_addr[0], VIDEO_PORT), video_seq, last_ts_ms, packet)
            video_seq = (video_seq + num_sent) % (2 ** 16)

    container.close()
    print(f"[服务端-推流] 文件 {video_path} 推流结束。")


# --- 主函数 (无变化) ---
def main():
    sockets = {'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
               'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
               'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)}
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
        manager.stop_stream(); [sock.close() for sock in sockets.values()]; print("[服务端] 服务器已关闭。")


if __name__ == "__main__":
    main()
