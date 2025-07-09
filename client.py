import cv2
import socket
import numpy as np
import threading
import time
import json
import pyaudio
from collections import defaultdict, deque
import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk

# 从共享配置文件中导入设置
from shared_config import *


# --- 1. 网络状态监控器 ---
class NetworkMonitor:
    """
    负责监控网络状态，主要是计算丢包率。
    """

    def __init__(self):
        self.received_packets = 0
        self.lost_packets = 0
        self.expected_frame_id = -1
        self.lock = threading.Lock()

    def record_packet(self, frame_id):
        """记录收到的每一个数据包（非FEC包）"""
        with self.lock:
            if self.expected_frame_id == -1:
                self.expected_frame_id = frame_id

            if frame_id > self.expected_frame_id:
                # 计算丢失的帧数
                self.lost_packets += frame_id - self.expected_frame_id
                self.expected_frame_id = frame_id + 1
            elif frame_id == self.expected_frame_id:
                self.expected_frame_id += 1

            self.received_packets += 1

    def get_statistics(self):
        """计算并返回丢包率"""
        with self.lock:
            total_packets = self.received_packets + self.lost_packets
            if total_packets == 0:
                return {"loss_rate": 0.0}

            loss_rate = self.lost_packets / total_packets
            # 重置计数器以便进行下一轮统计
            self.received_packets = 0
            self.lost_packets = 0
            return {"loss_rate": loss_rate}


# --- 2. 帧重组与视频抖动缓冲 ---
class VideoJitterBuffer:
    """
    一个更完善的帧缓冲和重组器。
    - 缓存乱序到达的数据包。
    - 使用FEC恢复丢失的数据包。
    - 实现一个抖动缓冲（Jitter Buffer）来平滑视频播放。
    - 按帧ID顺序输出帧。
    """

    def __init__(self, buffer_time_ms=100):
        self.packet_buffers = defaultdict(lambda: {
            "packets": {}, "fec_packets": {}, "total_packets": -1,
            "received_count": 0, "timestamp": time.time()
        })
        self.ready_queue = deque()  # 存储已解码的完整帧 (frame_id, frame_data)
        self.lock = threading.Lock()
        self.buffer_delay = buffer_time_ms / 1000.0
        self.last_played_frame_id = -1

    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        """添加收到的UDP包到缓冲区"""
        if len(packet) < 9: return  # 包头不完整

        frame_id = int.from_bytes(packet[:4], 'big')
        is_fec = int.from_bytes(packet[8:9], 'big') == 1

        with self.lock:
            # 如果帧已经太旧，则直接丢弃
            if frame_id <= self.last_played_frame_id:
                return

            buffer = self.packet_buffers[frame_id]
            buffer["timestamp"] = time.time()

            if is_fec:
                packet_index = int.from_bytes(packet[4:6], 'big')
                buffer["fec_packets"][packet_index] = packet[9:]
            else:
                # 首次记录该帧的数据包时，更新网络监控器
                if buffer["total_packets"] == -1:
                    monitor.record_packet(frame_id)

                packet_index = int.from_bytes(packet[4:6], 'big')
                total_packets = int.from_bytes(packet[6:8], 'big')
                if packet_index not in buffer["packets"]:
                    buffer["packets"][packet_index] = packet[9:]
                    buffer["total_packets"] = total_packets
                    buffer["received_count"] += 1

            self._try_reassemble(frame_id)

    def _try_reassemble(self, frame_id):
        """尝试重组帧，包括FEC恢复"""
        buffer = self.packet_buffers[frame_id]

        # 检查是否所有数据包都已到达
        if buffer["total_packets"] != -1 and buffer["received_count"] == buffer["total_packets"]:
            self._push_to_ready_queue(frame_id)
            return

        # 尝试使用FEC包进行恢复
        for fec_index, fec_data in buffer["fec_packets"].items():
            start_idx = fec_index * FEC_GROUP_SIZE
            end_idx = start_idx + FEC_GROUP_SIZE

            missing_indices = []
            group_packets_data = []
            for i in range(start_idx, end_idx):
                if i < buffer["total_packets"]:
                    if i in buffer["packets"]:
                        group_packets_data.append(buffer["packets"][i])
                    else:
                        missing_indices.append(i)

            # 如果只有一个包丢失，则可以恢复
            if len(missing_indices) == 1:
                missing_idx = missing_indices[0]
                print(f"[FEC] 正在恢复帧 {frame_id} 的数据包 {missing_idx}")

                # 异或恢复
                max_len = max(len(d) for d in group_packets_data + [fec_data])
                recovered_data = bytearray(max_len)
                all_data = group_packets_data + [fec_data]
                for d in all_data:
                    padded_d = d.ljust(max_len, b'\0')
                    for i in range(max_len):
                        recovered_data[i] ^= padded_d[i]

                buffer["packets"][missing_idx] = bytes(recovered_data)
                buffer["received_count"] += 1

                if buffer["received_count"] == buffer["total_packets"]:
                    self._push_to_ready_queue(frame_id)
                    return

    def _push_to_ready_queue(self, frame_id):
        """将重组好的帧放入待播放队列"""
        if frame_id not in self.packet_buffers: return

        buffer = self.packet_buffers.pop(frame_id)
        sorted_packets = [buffer["packets"][i] for i in sorted(buffer["packets"].keys())]
        frame_data = b"".join(sorted_packets)

        # 将 (frame_id, frame_data) 元组插入到已排序的队列中
        self.ready_queue.append((frame_id, frame_data))
        self.ready_queue = deque(sorted(self.ready_queue))

    def get_frame(self):
        """从抖动缓冲中获取一帧用于显示"""
        with self.lock:
            if not self.ready_queue:
                return None

            # 检查是否有更旧的帧需要清理
            self.cleanup_old_frames()

            frame_id, frame_data = self.ready_queue.popleft()
            self.last_played_frame_id = frame_id
            return frame_data

    def cleanup(self):
        """清理过时的包缓冲区"""
        with self.lock:
            cutoff_time = time.time() - 3  # 清理3秒前的旧缓冲区
            old_frame_ids = [fid for fid, buf in self.packet_buffers.items() if buf["timestamp"] < cutoff_time]
            for fid in old_frame_ids:
                del self.packet_buffers[fid]

    def cleanup_old_frames(self):
        """清理ready_queue中过时的帧，避免内存堆积"""
        # 假设我们只保留最近的几帧，或者根据时间戳清理
        # 这里简单地清理比当前播放帧ID更旧的帧，或者当队列过长时清理
        while len(self.ready_queue) > 20:  # 保持最多20帧在队列中
            self.ready_queue.popleft()


# --- 3. 音频抖动缓冲 ---
class AudioJitterBuffer:
    """
    一个简单的音频抖动缓冲，用于平滑音频播放并降低延迟。
    """

    def __init__(self, max_size=5):  # 缓冲最多5个音频包
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_chunk(self, chunk):
        with self.lock:
            self.queue.append(chunk)

    def get_chunk(self):
        with self.lock:
            if not self.queue:
                return None
            return self.queue.popleft()


# --- 4. 媒体处理线程 ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag, status_ref):
    """接收视频数据包的线程"""
    while running_flag['running']:
        try:
            data, _ = sock.recvfrom(65535)
            jitter_buffer.add_packet(data, monitor)
            status_ref['video_active'] = True  # 标记视频流活跃
        except socket.error:
            print("[Video] 套接字错误，接收线程退出。")
            status_ref['video_active'] = False  # 标记视频流不活跃
            break


def audio_receiver_thread(sock, audio_buffer, running_flag):
    """接收音频包并放入抖动缓冲"""
    while running_flag['running']:
        try:
            data, _ = sock.recvfrom(AUDIO_CHUNK * 2)
            audio_buffer.add_chunk(data)
        except socket.error:
            print("[Audio] 套接字错误，音频接收线程退出。")
            break


def audio_player_thread(audio_buffer, running_flag):
    """从抖动缓冲中获取并播放音频数据的线程"""
    p = pyaudio.PyAudio()
    try:
        audio_stream = p.open(format=AUDIO_FORMAT,
                              channels=AUDIO_CHANNELS,
                              rate=AUDIO_RATE,
                              output=True,
                              frames_per_buffer=AUDIO_CHUNK)  # 关键：设置与服务器匹配的块大小
    except Exception as e:
        print(f"[Audio] 无法打开音频播放设备: {e}")
        return

    print("[Audio] 音频播放已准备就绪。")
    while running_flag['running']:
        chunk = audio_buffer.get_chunk()
        if chunk:
            try:
                audio_stream.write(chunk)
            except IOError as e:
                print(f"[Audio] 播放音频时出错: {e}")
        else:
            time.sleep(0.01)  # 缓冲为空时短暂等待

    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()
    print("[Audio] 音频播放已停止。")


def display_thread(video_jitter_buffer, running_flag, status_ref, video_label, root, scaling_mode, target_size):
    """
    显示视频帧的线程。
    - 使用事件驱动的尺寸更新，解决“渐进式”缩放问题。
    - 使用最快的OpenCV插值算法，确保低延迟。
    """
    last_frame_time = time.time()
    WAIT_THRESHOLD = 1.0

    def update_frame():
        nonlocal last_frame_time
        if not running_flag['running']:
            return

        frame_data = video_jitter_buffer.get_frame()

        if frame_data:
            last_frame_time = time.time()
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # --- 高性能视频缩放逻辑 (使用OpenCV) ---
                    mode = scaling_mode.get()
                    # 使用<Configure>事件更新的尺寸，避免在循环中频繁调用winfo_*
                    label_w = target_size['w']
                    label_h = target_size['h']

                    if label_w > 1 and label_h > 1:
                        if mode == "fill":
                            # 模式1: 拉伸填充 - 使用最快的cv2.INTER_NEAREST插值算法
                            frame = cv2.resize(frame, (label_w, label_h), interpolation=cv2.INTER_NEAREST)
                        elif mode == "fit":
                            # 模式2: 按比例缩放 - 使用最快的cv2.INTER_NEAREST插值算法
                            h, w, _ = frame.shape
                            ratio = min(label_w / w, label_h / h)
                            new_size = (int(w * ratio), int(h * ratio))
                            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_NEAREST)
                        # 模式3: "original" - 无需操作

                    # 转换到Tkinter格式
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)

                    video_label.config(image=img_tk)
                    video_label.image = img_tk
                    status_ref['video_active'] = True
                    if "正在接收" not in status_label.cget("text"):
                        status_label.config(text="状态: 正在接收视频流...", fg="green")
                else:
                    if time.time() - last_frame_time > WAIT_THRESHOLD:
                        status_label.config(text="状态: 解码失败", fg="red")
            except Exception as e:
                print(f"[Display] 解码或显示帧时出错: {e}")
        else:
            if time.time() - last_frame_time > WAIT_THRESHOLD:
                status_ref['video_active'] = False
                status_label.config(text="状态: 等待视频流...", fg="orange")

        root.after(15, update_frame)

    root.after(15, update_frame)


# --- 5. 控制信令发送 ---
def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    """定期向服务器发送网络状态反馈的线程"""
    while running_flag['running']:
        time.sleep(1)  # 每秒发送一次反馈
        stats = monitor.get_statistics()
        try:
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error as e:
            print(f"[Feedback] 发送反馈失败: {e}")


# --- 6. 主函数 ---
def main():
    # Tkinter GUI 设置
    root = tk.Tk()
    root.title("实时视频流客户端")
    root.geometry("800x600")
    root.resizable(True, True)
    root.configure(bg="#2c3e50")

    # 视频显示区域
    video_label = tk.Label(root, bg="#000000", bd=2, relief="sunken")
    video_label.pack(pady=10, padx=10, fill="both", expand=True)

    target_size = {'w': 0, 'h': 0}
    def on_resize(event):
        target_size['w'] = event.width
        target_size['h'] = event.height
    video_label.bind("<Configure>", on_resize)

    placeholder_img = Image.new('RGB', (640, 480), color='gray')
    placeholder_tk = ImageTk.PhotoImage(image=placeholder_img)
    video_label.config(image=placeholder_tk)
    video_label.image = placeholder_tk

    scaling_mode = tk.StringVar(value="fit")
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_radiobutton(label="自适应缩放 (保持宽高比)", variable=scaling_mode, value="fit")
    context_menu.add_radiobutton(label="拉伸填充 (忽略宽高比)", variable=scaling_mode, value="fill")
    context_menu.add_radiobutton(label="原始大小", variable=scaling_mode, value="original")
    def show_context_menu(event):
        context_menu.post(event.x_root, event.y_root)
    video_label.bind("<Button-3>", show_context_menu)

    input_frame = tk.Frame(root, bg="#2c3e50")
    input_frame.pack(pady=5)
    ip_label = tk.Label(input_frame, text="服务器IP:", bg="#2c3e50", fg="white", font=("Arial", 12))
    ip_label.pack(side=tk.LEFT, padx=5)
    server_ip_entry = tk.Entry(input_frame, width=30, font=("Arial", 12), bd=2, relief="groove")
    server_ip_entry.insert(0, "127.0.0.1")
    server_ip_entry.pack(side=tk.LEFT, padx=5)
    connect_button = tk.Button(input_frame, text="连接", font=("Arial", 12, "bold"), bg="#3498db", fg="white", activebackground="#2980b9", relief="raised", bd=3)
    connect_button.pack(side=tk.LEFT, padx=5)

    global status_label
    status_label = tk.Label(root, text="状态: 请输入服务器IP并点击连接", bg="#2c3e50", fg="white", font=("Arial", 10))
    status_label.pack(pady=5)

    # --- 提升变量作用域以便在on_closing中访问 ---
    running_flag = {'running': True}
    sockets = {'video': None, 'audio': None, 'control': None}
    threads = {'video_recv': None, 'audio_recv': None, 'audio_play': None, 'feedback': None, 'cleanup': None}

    def start_client_threads(server_ip):
        nonlocal running_flag
        
        if not running_flag['running']:
            running_flag['running'] = True

        server_address = (server_ip, CONTROL_PORT)
        
        try:
            sockets['video'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets['video'].bind(('', VIDEO_PORT))
            sockets['audio'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets['audio'].bind(('', AUDIO_PORT))
            sockets['control'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            messagebox.showerror("网络错误", f"无法绑定端口: {e}")
            return

        status_label.config(text=f"状态: 正在连接到 {server_ip}...", fg="blue")
        connect_button.config(state=tk.DISABLED)

        monitor = NetworkMonitor()
        video_jitter_buffer = VideoJitterBuffer(buffer_time_ms=150)
        audio_jitter_buffer = AudioJitterBuffer(max_size=5)
        status_ref = {'video_active': False}

        threads['video_recv'] = threading.Thread(target=video_receiver_thread, args=(sockets['video'], video_jitter_buffer, monitor, running_flag, status_ref))
        threads['audio_recv'] = threading.Thread(target=audio_receiver_thread, args=(sockets['audio'], audio_jitter_buffer, running_flag))
        threads['audio_play'] = threading.Thread(target=audio_player_thread, args=(audio_jitter_buffer, running_flag))
        threads['feedback'] = threading.Thread(target=feedback_sender_thread, args=(sockets['control'], server_address, monitor, running_flag))
        threads['cleanup'] = threading.Thread(target=lambda: (time.sleep(5), video_jitter_buffer.cleanup()), daemon=True)

        display_thread(video_jitter_buffer, running_flag, status_ref, video_label, root, scaling_mode, target_size)

        for thread in threads.values():
            if thread:
                thread.daemon = True
                thread.start()
        
        try:
            sockets['control'].sendto(json.dumps({"status": "connect"}).encode(), server_address)
            status_label.config(text="状态: 连接成功，等待视频流...", fg="green")
        except socket.error as e:
            messagebox.showerror("连接错误", f"无法发送连接请求: {e}")
            status_label.config(text="状态: 连接失败", fg="red")
            running_flag['running'] = False
        
        connect_button.config(state=tk.NORMAL)

    connect_button.config(command=lambda: start_client_threads(server_ip_entry.get()))

    def on_closing():
        print("正在关闭客户端...")
        if not running_flag['running']:
            root.destroy()
            return

        running_flag['running'] = False
        
        # 等待线程结束
        for name, thread in threads.items():
            if thread and thread.is_alive():
                print(f"正在等待 {name} 线程结束...")
                thread.join(timeout=0.5)

        # 关闭套接字
        for name, sock in sockets.items():
            if sock:
                print(f"正在关闭 {name} 套接字...")
                sock.close()

        print("所有资源已释放。")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
