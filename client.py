# client.py

import cv2
import socket
import numpy as np
import threading
import time
import json
from collections import defaultdict, deque
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pyaudio

from shared_config import *

# --- 辅助类 (JitterBuffer, NetworkMonitor) ---
class NetworkMonitor:
    def __init__(self): self.reset()
    def record_packet(self, frame_id):
        with self.lock:
            if self.expected_frame_id == -1: self.expected_frame_id = frame_id
            if frame_id > self.expected_frame_id: self.lost_packets += frame_id - self.expected_frame_id
            self.expected_frame_id = frame_id + 1; self.received_packets += 1
    def get_statistics(self):
        with self.lock:
            total = self.received_packets + self.lost_packets
            loss_rate = self.lost_packets / total if total > 0 else 0.0
            self.received_packets, self.lost_packets = 0, 0
            return {"loss_rate": loss_rate}
    def reset(self): self.lock = threading.Lock(); self.received_packets, self.lost_packets, self.expected_frame_id = 0, 0, -1

class VideoJitterBuffer:
    def __init__(self): self.reset()
    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        if len(packet) < 8: return
        frame_id, packet_index, total_packets = int.from_bytes(packet[:4], 'big'), int.from_bytes(packet[4:6], 'big'), int.from_bytes(packet[6:8], 'big')
        with self.lock:
            if frame_id <= self.last_played_frame_id: return
            if packet_index in self.packet_buffers[frame_id]["packets"]: return
            if not self.packet_buffers[frame_id]["packets"]: monitor.record_packet(frame_id)
            buffer = self.packet_buffers[frame_id]
            buffer["packets"][packet_index], buffer["total_packets"] = packet[8:], total_packets
            if len(buffer["packets"]) == buffer["total_packets"]: self._push_to_ready_queue(frame_id)
    def _push_to_ready_queue(self, frame_id):
        buffer = self.packet_buffers.pop(frame_id)
        data = b"".join(buffer["packets"][i] for i in sorted(buffer["packets"]))
        self.ready_queue.append((frame_id, data)); self.ready_queue = deque(sorted(self.ready_queue))
    def get_frame(self):
        with self.lock:
            if not self.ready_queue: return None
            frame_id, data = self.ready_queue.popleft()
            self.last_played_frame_id = frame_id
            return frame_id, data
    def reset(self): self.lock = threading.Lock(); self.packet_buffers = defaultdict(lambda: {"packets": {}, "total_packets": -1}); self.ready_queue = deque(); self.last_played_frame_id = -1

class AudioJitterBuffer:
    def __init__(self, max_size=30): self.queue = deque(maxlen=max_size); self.lock = threading.Lock()
    def add_chunk(self, chunk): self.queue.append(chunk)
    def get_chunk(self): return self.queue.popleft() if self.queue else None
    def clear(self): self.queue.clear()


# --- 媒体处理线程 ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag):
    while running_flag.get('running'):
        try:
            data, _ = sock.recvfrom(65535); jitter_buffer.add_packet(data, monitor)
        except socket.error: break

def audio_receiver_thread(sock, audio_buffer, running_flag):
    while running_flag.get('running'):
        try:
            data, _ = sock.recvfrom(4096); audio_buffer.add_chunk(data)
        except socket.error: break

def audio_player_thread(audio_buffer, running_flag):
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, output=True, frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[Audio] PyAudio Error: {e}"); p.terminate(); return

    print("[Audio] Player is running.")
    while running_flag.get('running'):
        chunk = audio_buffer.get_chunk()
        if chunk and stream: stream.write(chunk)
        else: time.sleep(0.005) # 短暂休眠避免空转

    if stream: stream.stop_stream(); stream.close()
    p.terminate()
    print("[Audio] Player stopped.")

def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    while running_flag.get('running'):
        time.sleep(1)
        if not running_flag.get('running'): break
        try:
            stats = monitor.get_statistics(); stats['command'] = 'heartbeat'
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error: break

# --- GUI和主逻辑 ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("实时视频流客户端")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # 状态变量
        self.is_connected = False
        self.running_flag = {'running': False}
        self.sockets = {}
        self.threads = {}
        self.server_address = None
        self.current_source = "无"
        self.scale_mode = tk.StringVar(value="fit") # 'adapt', 'fit', 'original'

        # 媒体处理
        self.monitor = NetworkMonitor()
        self.video_jitter_buffer = VideoJitterBuffer()
        self.audio_jitter_buffer = AudioJitterBuffer()
        self.last_frame_pil = None # 存储最近解码的PIL Image

        self._create_widgets()
        self.start_display_updater() # 启动独立的UI更新循环

    def _create_widgets(self):
        # 连接栏
        conn_frame = tk.Frame(self.root, bg="#34495e"); conn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(conn_frame, text="服务器IP:", bg="#34495e", fg="white").pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(conn_frame, width=20); self.ip_entry.insert(0, "127.0.0.1"); self.ip_entry.pack(side=tk.LEFT, padx=5)
        self.connect_btn = tk.Button(conn_frame, text="连接", command=self.toggle_connection); self.connect_btn.pack(side=tk.LEFT, padx=5)

        # 主内容区
        main_frame = tk.Frame(self.root, bg="#2c3e50"); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 播放列表
        list_frame = tk.Frame(main_frame, bg="#34495e"); list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tk.Label(list_frame, text="播放列表", bg="#34495e", fg="white").pack()
        self.video_listbox = tk.Listbox(list_frame, bg="#ecf0f1", selectbackground="#3498db"); self.video_listbox.pack(fill=tk.Y, expand=True)
        self.play_btn = tk.Button(list_frame, text="播放选中项", command=self.play_selected, state=tk.DISABLED); self.play_btn.pack(fill=tk.X, pady=5)

        # 视频显示区域
        self.video_label = tk.Label(main_frame, bg="black"); self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_label = tk.Label(self.root, text="状态: 未连接", bd=1, relief=tk.SUNKEN, anchor=tk.W); self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 右键菜单
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_radiobutton(label="自适应缩放 (Adapt)", variable=self.scale_mode, value="adapt", command=self.on_scale_mode_change)
        self.context_menu.add_radiobutton(label="按比例缩放 (Fit)", variable=self.scale_mode, value="fit", command=self.on_scale_mode_change)
        self.context_menu.add_radiobutton(label="原始大小 (Original)", variable=self.scale_mode, value="original", command=self.on_scale_mode_change)
        self.video_label.bind("<Button-3>", lambda e: self.context_menu.post(e.x_root, e.y_root))
        # 窗口大小改变时也触发重绘
        self.video_label.bind("<Configure>", self.on_scale_mode_change)

    # --- 连接与控制 ---
    def toggle_connection(self):
        if self.is_connected: self.disconnect()
        else: self.connect()

    def connect(self):
        server_ip = self.ip_entry.get()
        if not server_ip: return messagebox.showerror("错误", "请输入服务器IP地址")

        self.server_address = (server_ip, CONTROL_PORT)
        self.running_flag['running'] = True
        try:
            self.sockets['control'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['video'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sockets['video'].bind(('', VIDEO_PORT))
            self.sockets['audio'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); self.sockets['audio'].bind(('', AUDIO_PORT))

            self.sockets['control'].settimeout(5)
            self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048)
            self.sockets['control'].settimeout(None)

            self.video_listbox.delete(0, tk.END)
            for item in json.loads(data.decode()): self.video_listbox.insert(tk.END, item)

            self.start_threads()
            self.is_connected = True
            self.connect_btn.config(text="断开")
            self.play_btn.config(state=tk.NORMAL)
            self.status_label.config(text="状态: 连接成功，请选择播放项")
        except Exception as e:
            messagebox.showerror("连接失败", str(e)); self.cleanup()

    def disconnect(self):
        if self.sockets.get('control') and self.server_address:
            try: self.sockets['control'].sendto(json.dumps({"command": "stop"}).encode(), self.server_address)
            except socket.error: pass
        self.cleanup()

    def play_selected(self):
        if not (selection := self.video_listbox.curselection()): return messagebox.showwarning("提示", "请选择一个项目")
        self.current_source = self.video_listbox.get(selection[0])
        self.video_jitter_buffer.reset(); self.audio_jitter_buffer.clear(); self.monitor.reset()
        self.sockets['control'].sendto(json.dumps({"command": "play", "source": self.current_source}).encode(), self.server_address)
        self.status_label.config(text=f"状态: 请求播放 {self.current_source}...")

    # --- 线程管理 ---
    def start_threads(self):
        thread_map = {
            'video_recv': (video_receiver_thread, (self.sockets['video'], self.video_jitter_buffer, self.monitor, self.running_flag)),
            'audio_recv': (audio_receiver_thread, (self.sockets['audio'], self.audio_jitter_buffer, self.running_flag)),
            'audio_play': (audio_player_thread, (self.audio_jitter_buffer, self.running_flag)),
            'feedback': (feedback_sender_thread, (self.sockets['control'], self.server_address, self.monitor, self.running_flag)),
        }
        for name, (target, args) in thread_map.items():
            self.threads[name] = threading.Thread(target=target, args=args, daemon=True); self.threads[name].start()
            print(f"[App] Thread '{name}' started.")

    # --- UI 更新与缩放 ---
    def start_display_updater(self):
        """主循环，负责从抖动缓冲区取帧并解码"""
        frame_tuple = self.video_jitter_buffer.get_frame()
        if frame_tuple:
            _, frame_data = frame_tuple
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.last_frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.redraw_video() # 解码后立即重绘
            except Exception as e:
                print(f"[Display] 解码错误: {e}")

        self.root.after(15, self.start_display_updater) # ~66 FPS 刷新率

    def on_scale_mode_change(self, event=None):
        """当缩放模式改变或窗口大小改变时，重绘最后一帧"""
        self.redraw_video()

    def redraw_video(self):
        """根据当前缩放模式，将最后一帧图像绘制到标签上"""
        if not self.last_frame_pil: return

        label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if label_w < 2 or label_h < 2: return # 窗口尚未完全渲染

        img = self.last_frame_pil
        mode = self.scale_mode.get()

        if mode == 'original':
            # 原始大小，直接使用
            img_resized = img
        elif mode == 'adapt':
            # 自适应缩放，拉伸填满
            img_resized = img.resize((label_w, label_h), Image.Resampling.LANCZOS)
        else: # 'fit'
            # 按比例缩放
            img_ratio = img.width / img.height
            label_ratio = label_w / label_h
            if label_ratio > img_ratio: # 窗口更宽，以高度为准
                new_h = label_h
                new_w = int(new_h * img_ratio)
            else: # 窗口更高或比例相同，以宽度为准
                new_w = label_w
                new_h = int(new_w / img_ratio)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建一个黑色背景，将缩放后的图像居中粘贴
        bg = Image.new('RGB', (label_w, label_h), 'black')
        paste_x = (label_w - img_resized.width) // 2
        paste_y = (label_h - img_resized.height) // 2
        bg.paste(img_resized, (paste_x, paste_y))

        img_tk = ImageTk.PhotoImage(image=bg)
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk

    # --- 清理 ---
    def cleanup(self):
        if not self.running_flag.get('running'): return
        print("[App] Cleaning up...")
        self.running_flag['running'] = False
        time.sleep(0.1)
        for sock in self.sockets.values(): sock.close()
        for thread in self.threads.values():
            if thread.is_alive(): thread.join(timeout=0.5)

        self.threads.clear(); self.sockets.clear()
        self.is_connected = False; self.current_source = "无"; self.last_frame_pil = None

        self.connect_btn.config(text="连接"); self.play_btn.config(state=tk.DISABLED)
        self.video_listbox.delete(0, tk.END)
        self.status_label.config(text="状态: 未连接")

        placeholder = ImageTk.PhotoImage(Image.new('RGB', (1, 1), 'black'))
        self.video_label.config(image=placeholder); self.video_label.image = placeholder
        self.video_jitter_buffer.reset(); self.audio_jitter_buffer.clear(); self.monitor.reset()

    def on_closing(self):
        self.disconnect(); self.root.destroy()

# --- 主函数 ---
def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()