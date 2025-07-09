import io
import cv2
import socket
import time
import threading
import json
import os
import av
import traceback

from shared_config import *

# --- 1. 视频文件扫描 ---
def get_video_files(path="videos"):
    if not os.path.exists(path):
        print(f"警告: 视频目录 '{path}' 不存在。将创建一个空目录。")
        os.makedirs(path)
        return []
    supported_formats = ('.mp4', '.mkv', '.avi', '.mov')
    return [f for f in os.listdir(path) if f.endswith(supported_formats)]

# --- 2. 自适应流控器 (无变化) ---
class AdaptiveStreamController:
    def __init__(self):
        self.configs = {
            "good": {"resolution": (640, 480), "jpeg_quality": 85, "fec_enabled": False, "fps_limit": 30},
            "medium": {"resolution": (480, 360), "jpeg_quality": 60, "fec_enabled": True, "fps_limit": 20},
            "poor": {"resolution": (320, 240), "jpeg_quality": 40, "fec_enabled": True, "fps_limit": 15},
        }
        self.current_strategy_name = "good"
        self.lock = threading.Lock()

    def get_current_strategy(self):
        with self.lock:
            return self.configs[self.current_strategy_name]

    def update_strategy(self, loss_rate):
        new_strategy_name = "good"
        if loss_rate >= 0.1: new_strategy_name = "poor"
        elif loss_rate >= 0.03: new_strategy_name = "medium"
        with self.lock:
            if self.current_strategy_name != new_strategy_name:
                self.current_strategy_name = new_strategy_name
                print(f"[Controller] 丢包率: {loss_rate:.2%}, 切换策略至: {new_strategy_name.upper()}")

# --- 3. 推流管理器 ---
class StreamerManager:
    """管理当前的推流线程，确保同一时间只有一个流在运行"""
    def __init__(self, video_sock, audio_sock, controller):
        self.video_sock = video_sock
        self.audio_sock = audio_sock
        self.controller = controller
        self.current_stream_thread = None
        self.running_flag = None
        self.lock = threading.Lock() # 这是一个非重入锁

    def _stop_stream_nolock(self):
        """停止当前流的内部方法，不获取锁。假定调用者已经持有锁。"""
        if self.running_flag:
            self.running_flag['running'] = False

        # 保存对旧线程的引用，以便在锁外等待它
        thread_to_join = self.current_stream_thread

        if thread_to_join and thread_to_join.is_alive():
            print("[Manager] 正在停止当前推流...")
            # 在锁内设置完标志后，在锁外等待，避免长时间持有锁
            thread_to_join.join(timeout=1)

        self.current_stream_thread = None
        # 避免在join后打印，因为可能超时了但线程仍在运行
        if not thread_to_join or not thread_to_join.is_alive():
            print("[Manager] 推流已停止。")
        else:
            print("[Manager] 停止推流超时，线程可能仍在后台退出。")


    def start_stream(self, source, client_addr):
        with self.lock: # 获取一次锁，保护整个“先停后启”操作
            # 1. 调用无锁的停止方法
            self._stop_stream_nolock()

            # 2. 现在安全地启动新流
            self.running_flag = {'running': True}

            if source == "camera":
                print("[Manager] 启动摄像头直播...")
                self.current_stream_thread = threading.Thread(
                    target=stream_from_camera,
                    args=(self.video_sock, self.audio_sock, self.controller, self.running_flag, client_addr),
                    daemon=True
                )
            else:
                video_path = os.path.join("videos", source)
                if os.path.exists(video_path):
                    print(f"[Manager] 启动文件点播: {source}")
                    self.current_stream_thread = threading.Thread(
                        target=stream_from_file,
                        args=(self.video_sock, self.audio_sock, self.controller, self.running_flag, client_addr, video_path),
                        daemon=True
                    )
                else:
                    print(f"[Manager] 错误: 找不到视频文件 {video_path}")
                    return

            self.current_stream_thread.start()

    def stop_stream(self):
        """公共的停止方法，供外部调用（例如，客户端断开连接时）"""
        with self.lock:
            self._stop_stream_nolock()

# --- 4. 控制信道处理 (已修正) ---
def control_channel_handler(sock, manager, client_address_ref, shutdown_event):
    video_list = get_video_files()
    print(f"[Control] 可用视频文件: {video_list}")
    while shutdown_event.is_set():
        try:
            sock.settimeout(1.0)
            data, addr = sock.recvfrom(1024)
            if client_address_ref['addr'] is None:
                print(f"[Control] 收到来自 {addr} 的连接请求。")
                client_address_ref['addr'] = addr
            message = json.loads(data.decode())
            command = message.get("command")
            if command == "get_list":
                print(f"[Control] 收到来自 {addr} 的视频列表请求。")
                response = json.dumps(["camera"] + video_list)
                sock.sendto(response.encode(), addr)
            elif command == "play":
                source = message.get("source")
                print(f"[Control] 收到来自 {addr} 的播放请求: {source}")
                if source:
                    manager.start_stream(source, addr)
            elif command == "stop":
                print(f"[Control] 收到来自 {addr} 的停止请求。")
                manager.stop_stream()
            elif 'loss_rate' in message:
                manager.controller.update_strategy(message['loss_rate'])
        except socket.timeout:
            continue
        except (ConnectionResetError, socket.error):
            if not shutdown_event.is_set():
                print(f"[Control] 套接字已关闭，线程正常退出。")
            else:
                print(f"[Control] 客户端 {client_address_ref.get('addr')} 断开连接。")
                manager.stop_stream()
                client_address_ref['addr'] = None
            break
        except Exception as e:
            print(f"[Control] 处理控制信令时发生错误: {e}")
    print("[Control] 控制信道处理器已停止。")

# --- 5. 推流实现 ---
def stream_from_camera(video_sock, audio_sock, controller, running_flag, client_addr):
    """从摄像头和麦克风捕获并推流 (健壮版)"""
    try:
        print("[Streamer] 摄像头推流功能已激活。")
        cap = cv2.VideoCapture(0)

        # --- CRITICAL FIX: 检查摄像头是否成功打开 ---
        if not cap.isOpened():
            print("[Streamer] 错误: 无法打开摄像头 0。请检查摄像头是否被其他程序占用。")
            return

        frame_id = 0
        video_dest = (client_addr[0], VIDEO_PORT)

        while running_flag['running']:
            ret, frame = cap.read()
            if not ret:
                print("[Streamer] 无法从摄像头读取帧，推流结束。")
                break

            strategy = controller.get_current_strategy()
            frame = cv2.resize(frame, strategy["resolution"])
            _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), strategy["jpeg_quality"]])
            frame_data = encoded_frame.tobytes()

            total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
            for i in range(total_packets):
                chunk = frame_data[i*PACKET_SIZE : (i+1)*PACKET_SIZE]
                header = frame_id.to_bytes(4, 'big') + i.to_bytes(2, 'big') + total_packets.to_bytes(2, 'big') + (0).to_bytes(1, 'big')
                video_sock.sendto(header + chunk, video_dest)

            frame_id = (frame_id + 1) % (2**32 - 1)
            time.sleep(1 / strategy["fps_limit"])
    except Exception as e:
        print(f"[Streamer-CAMERA] 摄像头推流线程发生严重错误: {e}")
        traceback.print_exc()
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("[Streamer] 摄像头推流结束。")

def stream_from_file(video_sock, audio_sock, controller, running_flag, client_addr, video_path):
    """使用PyAV从视频文件推流（健壮版）"""
    container = None
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0] if container.streams.video else None
        audio_stream = container.streams.audio[0] if container.streams.audio else None

        if not video_stream:
            print(f"[Streamer] 错误: 文件 '{video_path}' 中没有找到可用的视频流。")
            return

        streams_to_demux = [s for s in [video_stream, audio_stream] if s is not None]
        print(f"[Streamer] 开始推流文件: {video_path}")
        print(f"           -> 客户端地址: {client_addr[0]}:{VIDEO_PORT}(V)/{AUDIO_PORT}(A)")
        print(f"           -> 视频流: {'存在' if video_stream else '不存在'}")
        print(f"           -> 音频流: {'存在' if audio_stream else '不存在'}")

        start_time = time.time()
        frame_id = 0
        video_dest = (client_addr[0], VIDEO_PORT)
        audio_dest = (client_addr[0], AUDIO_PORT)

        for packet in container.demux(*streams_to_demux):
            if not running_flag['running']:
                break

            if packet.time_base is not None:
                current_pts = float(packet.pts * packet.time_base)
                elapsed_time = time.time() - start_time
                if current_pts > elapsed_time:
                    time.sleep(current_pts - elapsed_time)

            if video_stream and packet.stream == video_stream:
                for frame in packet.decode():
                    img = frame.to_image()
                    strategy = controller.get_current_strategy()
                    img = img.resize(strategy["resolution"])
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=strategy["jpeg_quality"])
                    frame_data = buffer.getvalue()
                    total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
                    for i in range(total_packets):
                        chunk = frame_data[i*PACKET_SIZE : (i+1)*PACKET_SIZE]
                        header = frame_id.to_bytes(4, 'big') + i.to_bytes(2, 'big') + total_packets.to_bytes(2, 'big') + (0).to_bytes(1, 'big')
                        video_sock.sendto(header + chunk, video_dest)
                    frame_id = (frame_id + 1) % (2**32 - 1)
            elif audio_stream and packet.stream == audio_stream:
                for frame in packet.decode():
                    audio_data = frame.to_ndarray().tobytes()
                    audio_sock.sendto(audio_data, audio_dest)

    except Exception as e:
        print(f"[Streamer-FILE] 推流线程 '{video_path}' 意外崩溃!")
        traceback.print_exc()
    finally:
        if container:
            container.close()
        print(f"[Streamer] 文件 '{video_path}' 推流结束或中止。")

# --- 6. 主函数 (已修正) ---
def main():
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    controller = AdaptiveStreamController()
    manager = StreamerManager(video_sock, audio_sock, controller)
    client_address_ref = {'addr': None}
    shutdown_event = threading.Event()
    shutdown_event.set()

    control_thread = threading.Thread(
        target=control_channel_handler,
        args=(control_sock, manager, client_address_ref, shutdown_event),
        daemon=True
    )

    try:
        control_sock.bind((SERVER_HOST, CONTROL_PORT))
        control_thread.start()
        print(f"[Server] 服务器已启动。在 {SERVER_HOST}:{CONTROL_PORT} 等待客户端连接...")
        print("[Server] 按下 Ctrl+C 关闭服务器。")
        while control_thread.is_alive():
            control_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[Server] 收到关闭信号 (Ctrl+C)...")
    except Exception as e:
        print(f"[Server] 主线程发生致命错误: {e}")
    finally:
        print("[Server] 开始关闭程序...")
        shutdown_event.clear()
        manager.stop_stream()
        control_sock.close()
        video_sock.close()
        audio_sock.close()
        if control_thread.is_alive():
            control_thread.join(timeout=2)
        print("[Server] 服务器已成功关闭。")

if __name__ == "__main__":
    main()