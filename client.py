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


def display_thread(video_jitter_buffer, running_flag, status_ref, video_label, root):
    """
    显示视频帧的线程，现在更新 Tkinter 标签。
    """

    def update_frame():
        if not running_flag['running']:
            return

        frame_data = video_jitter_buffer.get_frame()

        if frame_data:
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    # 将OpenCV BGR图像转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # 调整图像大小以适应标签 (如果需要，可以根据标签实际大小调整)
                    # 例如: img.thumbnail((video_label.winfo_width(), video_label.winfo_height()), Image.LANCZOS)

                    img_tk = ImageTk.PhotoImage(image=img)
                    video_label.config(image=img_tk)
                    video_label.image = img_tk  # 保持引用，防止被垃圾回收
                    status_ref['video_active'] = True
                    status_label.config(text="状态: 正在接收视频流...", fg="green")
                else:
                    status_ref['video_active'] = False
                    status_label.config(text="状态: 解码失败或无视频数据", fg="red")
            except Exception as e:
                print(f"[Display] 解码或显示帧时出错: {e}")
                status_ref['video_active'] = False
                status_label.config(text="状态: 显示错误", fg="red")
        else:
            status_ref['video_active'] = False
            status_label.config(text="状态: 等待视频流...", fg="orange")

        # 调度下一次更新
        root.after(10, update_frame)  # 每10毫秒更新一次

    # 启动第一次更新
    root.after(10, update_frame)


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
    root.geometry("800x600")  # 初始窗口大小
    root.resizable(True, True)  # 允许调整大小
    root.configure(bg="#2c3e50")  # 深色背景

    # 视频显示区域
    video_label = tk.Label(root, bg="#000000", bd=2, relief="sunken")
    video_label.pack(pady=10, padx=10, fill="both", expand=True)

    # 初始占位符图像
    placeholder_img = Image.new('RGB', (640, 480), color='gray')
    placeholder_tk = ImageTk.PhotoImage(image=placeholder_img)
    video_label.config(image=placeholder_tk)
    video_label.image = placeholder_tk  # 保持引用

    # IP 输入框和连接按钮
    input_frame = tk.Frame(root, bg="#2c3e50")
    input_frame.pack(pady=5)

    ip_label = tk.Label(input_frame, text="服务器IP:", bg="#2c3e50", fg="white", font=("Arial", 12))
    ip_label.pack(side=tk.LEFT, padx=5)

    server_ip_entry = tk.Entry(input_frame, width=30, font=("Arial", 12), bd=2, relief="groove")
    server_ip_entry.insert(0, "127.0.0.1")  # 默认本地IP
    server_ip_entry.pack(side=tk.LEFT, padx=5)

    connect_button = tk.Button(input_frame, text="连接", command=lambda: start_client_threads(server_ip_entry.get()),
                               font=("Arial", 12, "bold"), bg="#3498db", fg="white", activebackground="#2980b9",
                               relief="raised", bd=3)
    connect_button.pack(side=tk.LEFT, padx=5)

    # 状态显示
    global status_label  # 使其在 display_thread 中可访问
    status_label = tk.Label(root, text="状态: 请输入服务器IP并点击连接", bg="#2c3e50", fg="white", font=("Arial", 10))
    status_label.pack(pady=5)

    # 日志输出区域 (可选)
    # log_text = scrolledtext.ScrolledText(root, height=5, bg="#34495e", fg="white", font=("Arial", 9))
    # log_text.pack(pady=5, padx=10, fill="x")
    # def print_to_log(text):
    #     log_text.insert(tk.END, text + "\n")
    #     log_text.see(tk.END)
    # import sys
    # sys.stdout.write = print_to_log # 重定向print输出到日志框

    # 初始化核心组件，这些需要全局或通过参数传递
    monitor = NetworkMonitor()
    video_jitter_buffer = VideoJitterBuffer(buffer_time_ms=150)
    audio_jitter_buffer = AudioJitterBuffer(max_size=5)
    running_flag = {'running': True}
    status_ref = {'video_active': False}  # 用于在线程间共享视频活跃状态

    # 定义启动客户端线程的函数
    def start_client_threads(server_ip):
        nonlocal monitor, video_jitter_buffer, audio_jitter_buffer, running_flag, status_ref

        # 如果已经连接，先停止旧线程
        if running_flag['running'] == False:  # 检查是否已经停止或需要重新初始化
            # 重新初始化标志和缓冲区
            running_flag['running'] = True
            monitor = NetworkMonitor()
            video_jitter_buffer = VideoJitterBuffer(buffer_time_ms=150)
            audio_jitter_buffer = AudioJitterBuffer(max_size=5)
            status_ref = {'video_active': False}

        server_address = (server_ip, CONTROL_PORT)

        # 初始化套接字 (每次连接时重新创建，防止旧连接问题)
        try:
            video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            video_sock.bind(('', VIDEO_PORT))
            audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            audio_sock.bind(('', AUDIO_PORT))
            control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as e:
            messagebox.showerror("网络错误", f"无法绑定端口或创建套接字: {e}")
            status_label.config(text="状态: 网络初始化失败", fg="red")
            return

        status_label.config(text=f"状态: 正在连接到 {server_ip}...", fg="blue")
        connect_button.config(state=tk.DISABLED)  # 连接中禁用按钮

        # 启动视频接收线程
        video_recv_thread = threading.Thread(target=video_receiver_thread,
                                             args=(video_sock, video_jitter_buffer, monitor, running_flag, status_ref))
        video_recv_thread.daemon = True
        video_recv_thread.start()

        # 启动音频接收线程
        audio_recv_thread = threading.Thread(target=audio_receiver_thread,
                                             args=(audio_sock, audio_jitter_buffer, running_flag))
        audio_recv_thread.daemon = True
        audio_recv_thread.start()

        # 启动音频播放线程
        audio_play_thread = threading.Thread(target=audio_player_thread, args=(audio_jitter_buffer, running_flag))
        audio_play_thread.daemon = True
        audio_play_thread.start()

        # 启动视频显示线程 (现在由 Tkinter 的 after 方法驱动)
        display_thread(video_jitter_buffer, running_flag, status_ref, video_label, root)

        # 启动网络反馈发送线程
        feedback_thread = threading.Thread(target=feedback_sender_thread,
                                           args=(control_sock, server_address, monitor, running_flag))
        feedback_thread.daemon = True

        # 启动缓冲区清理线程
        cleanup_thread = threading.Thread(target=lambda: (
            time.sleep(5), video_jitter_buffer.cleanup()
        ), daemon=True)
        cleanup_thread.start()

        # 尝试发送连接请求
        try:
            control_sock.sendto(json.dumps({"status": "connect"}).encode(), server_address)
            feedback_thread.start()
            status_label.config(text="状态: 连接成功，等待视频流...", fg="green")
        except socket.error as e:
            messagebox.showerror("连接错误", f"无法发送连接请求到 {server_ip}: {e}")
            status_label.config(text="状态: 连接失败", fg="red")
            running_flag['running'] = False  # 停止所有线程
            connect_button.config(state=tk.NORMAL)  # 重新启用按钮
            return

        connect_button.config(state=tk.NORMAL)  # 连接成功后重新启用按钮，允许重新连接

    # 处理窗口关闭事件
    def on_closing():
        print("正在关闭客户端...")
        running_flag['running'] = False
        # 确保所有套接字关闭
        try:
            video_sock.close()
        except NameError:
            pass  # 如果未初始化，则跳过
        try:
            audio_sock.close()
        except NameError:
            pass
        try:
            control_sock.close()
        except NameError:
            pass

        # 等待线程结束 (可选，但有助于确保资源释放)
        # video_recv_thread.join(timeout=1)
        # audio_recv_thread.join(timeout=1)
        # audio_play_thread.join(timeout=1)
        # feedback_thread.join(timeout=1)

        root.destroy()  # 销毁 Tkinter 窗口

    root.protocol("WM_DELETE_WINDOW", on_closing)  # 捕获窗口关闭事件

    root.mainloop()  # 启动 Tkinter 事件循环


if __name__ == "__main__":
    main()
