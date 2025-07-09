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

# --- Classes (unchanged) ---
class NetworkMonitor:
    def __init__(self):
        self.received_packets, self.lost_packets, self.expected_frame_id = 0, 0, -1
        self.lock = threading.Lock()
    def record_packet(self, frame_id):
        with self.lock:
            if self.expected_frame_id == -1: self.expected_frame_id = frame_id
            if frame_id > self.expected_frame_id:
                self.lost_packets += frame_id - self.expected_frame_id
            self.expected_frame_id = frame_id + 1
            self.received_packets += 1
    def get_statistics(self):
        with self.lock:
            total = self.received_packets + self.lost_packets
            loss_rate = self.lost_packets / total if total > 0 else 0.0
            self.received_packets, self.lost_packets = 0, 0
            return {"loss_rate": loss_rate}

class VideoJitterBuffer:
    def __init__(self):
        self.packet_buffers = defaultdict(lambda: {"packets": {}, "total_packets": -1, "received_count": 0})
        self.ready_queue = deque()
        self.lock = threading.Lock()
        self.last_played_frame_id = -1
    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        if len(packet) < 9: return
        frame_id = int.from_bytes(packet[:4], 'big')
        with self.lock:
            if frame_id <= self.last_played_frame_id: return
            buffer = self.packet_buffers[frame_id]
            if buffer["total_packets"] == -1: monitor.record_packet(frame_id)
            packet_index = int.from_bytes(packet[4:6], 'big')
            if packet_index not in buffer["packets"]:
                buffer["packets"][packet_index] = packet[9:]
                buffer["total_packets"] = int.from_bytes(packet[6:8], 'big')
                buffer["received_count"] += 1
            if buffer["received_count"] > 0 and buffer["received_count"] == buffer["total_packets"]:
                self._push_to_ready_queue(frame_id)
    def _push_to_ready_queue(self, frame_id):
        if frame_id in self.packet_buffers:
            buffer = self.packet_buffers.pop(frame_id)
            frame_data = b"".join([buffer["packets"][i] for i in sorted(buffer["packets"].keys())])
            self.ready_queue.append((frame_id, frame_data))
            # Keep the queue sorted by frame_id
            self.ready_queue = deque(sorted(self.ready_queue, key=lambda item: item[0]))
    def get_frame(self):
        with self.lock:
            return self.ready_queue.popleft() if self.ready_queue else None
    def clear(self):
        with self.lock:
            self.packet_buffers.clear()
            self.ready_queue.clear()
            self.last_played_frame_id = -1

class AudioJitterBuffer:
    def __init__(self, max_size=20):
        self.queue = deque(maxlen=max_size)
        self.lock = threading.Lock()
    def add_chunk(self, chunk): self.queue.append(chunk)
    def get_chunk(self): return self.queue.popleft() if self.queue else None
    def clear(self): self.queue.clear()

# --- Media Threads (unchanged) ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag):
    while running_flag['running']:
        try:
            data, _ = sock.recvfrom(65535)
            jitter_buffer.add_packet(data, monitor)
        except socket.error:
            if running_flag['running']: print("[Video] Socket error in receiver.")
            break

def audio_receiver_thread(sock, audio_buffer, running_flag):
    while running_flag['running']:
        try:
            data, _ = sock.recvfrom(4096)
            audio_buffer.add_chunk(data)
        except socket.error:
            if running_flag['running']: print("[Audio] Socket error in receiver.")
            break

def audio_player_thread(audio_buffer, running_flag):
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, output=True, frames_per_buffer=AUDIO_CHUNK)
        while running_flag['running']:
            chunk = audio_buffer.get_chunk()
            if chunk:
                stream.write(chunk)
            else:
                time.sleep(0.01)
    except Exception as e:
        print(f"[Audio] PyAudio Error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

# --- Display Thread (REFACTORED) ---
def display_thread(app_instance):
    WAIT_THRESHOLD = 1.5

    # Use an instance variable on the App object to avoid scope issues.
    app_instance.last_frame_time = time.time()

    def update_frame():
        if not app_instance.running_flag['running']:
            return

        frame_tuple = app_instance.video_jitter_buffer.get_frame()
        if frame_tuple:
            frame_id, frame_data = frame_tuple
            app_instance.video_jitter_buffer.last_played_frame_id = frame_id
            app_instance.last_frame_time = time.time() # Update time upon receiving a frame

            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_tk = ImageTk.PhotoImage(image=img)
                    app_instance.video_label.config(image=img_tk)
                    app_instance.video_label.image = img_tk

                    if "正在播放" not in app_instance.status_label.cget("text"):
                        app_instance.status_label.config(text=f"状态: {app_instance.current_source} - 正在播放...")
            except Exception as e:
                print(f"[Display] Error decoding/displaying frame: {e}")

        if time.time() - app_instance.last_frame_time > WAIT_THRESHOLD and "等待媒体流" not in app_instance.status_label.cget("text"):
            app_instance.status_label.config(text=f"状态: {app_instance.current_source} - 等待媒体流...")

        app_instance.root.after(15, update_frame)

    app_instance.root.after(15, update_frame)

# --- Feedback Thread (unchanged) ---
def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    while running_flag['running']:
        time.sleep(1)
        stats = monitor.get_statistics()
        try:
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error:
            if running_flag['running']: print("[Feedback] Socket error.")
            break

# --- GUI and Main Logic ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("实时视频流客户端")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        self.is_connected = False
        self.running_flag = {'running': False}
        self.sockets = {'video': None, 'audio': None, 'control': None}
        self.threads = {}
        self.server_address = None
        self.current_source = "无"
        self.last_frame_time = 0 # NEW: Initialize instance variable

        self.monitor = NetworkMonitor()
        self.video_jitter_buffer = VideoJitterBuffer()
        self.audio_jitter_buffer = AudioJitterBuffer()

        self._create_widgets()

    def _create_widgets(self):
        # This function is unchanged
        conn_frame = tk.Frame(self.root, bg="#34495e")
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(conn_frame, text="服务器IP:", bg="#34495e", fg="white").pack(side=tk.LEFT, padx=5)
        self.ip_entry = tk.Entry(conn_frame, width=20)
        self.ip_entry.insert(0, "127.0.0.1")
        self.ip_entry.pack(side=tk.LEFT, padx=5)
        self.connect_btn = tk.Button(conn_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        list_frame = tk.Frame(main_frame, bg="#34495e")
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tk.Label(list_frame, text="播放列表", bg="#34495e", fg="white").pack()
        self.video_listbox = tk.Listbox(list_frame, bg="#ecf0f1", selectbackground="#3498db")
        self.video_listbox.pack(fill=tk.Y, expand=True)
        self.play_btn = tk.Button(list_frame, text="播放选中项", command=self.play_selected, state=tk.DISABLED)
        self.play_btn.pack(fill=tk.X, pady=5)
        self.video_label = tk.Label(main_frame, bg="black")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(self.root, text="状态: 未连接", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

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
            self.sockets['video'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['video'].bind(('', VIDEO_PORT))
            self.sockets['audio'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['audio'].bind(('', AUDIO_PORT))
            self.sockets['control'].settimeout(5)
            self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048)
            self.sockets['control'].settimeout(None)
            video_list = json.loads(data.decode())
            self.video_listbox.delete(0, tk.END)
            for item in video_list: self.video_listbox.insert(tk.END, item)
            self.start_threads()
            self.is_connected = True
            self.connect_btn.config(text="断开")
            self.play_btn.config(state=tk.NORMAL)
            self.status_label.config(text="状态: 连接成功，请选择播放项")
        except socket.timeout:
            messagebox.showerror("连接失败", "无法从服务器获取列表，请求超时。")
            self.cleanup()
        except Exception as e:
            messagebox.showerror("连接失败", str(e))
            self.cleanup()

    def disconnect(self):
        if self.sockets['control'] and self.server_address:
            try:
                self.sockets['control'].sendto(json.dumps({"command": "stop"}).encode(), self.server_address)
            except socket.error: pass
        self.cleanup()

    def play_selected(self):
        selection = self.video_listbox.curselection()
        if not selection: return messagebox.showwarning("提示", "请先从列表中选择一个项目")
        self.current_source = self.video_listbox.get(selection[0])
        self.video_jitter_buffer.clear()
        self.audio_jitter_buffer.clear()

        # Reset the timer when a new source is requested
        self.last_frame_time = time.time()

        play_command = {"command": "play", "source": self.current_source}
        self.sockets['control'].sendto(json.dumps(play_command).encode(), self.server_address)
        self.status_label.config(text=f"状态: 请求播放 {self.current_source}...")

    def start_threads(self):
        self.threads['video_recv'] = threading.Thread(target=video_receiver_thread, args=(self.sockets['video'], self.video_jitter_buffer, self.monitor, self.running_flag))
        self.threads['audio_recv'] = threading.Thread(target=audio_receiver_thread, args=(self.sockets['audio'], self.audio_jitter_buffer, self.running_flag))
        self.threads['audio_play'] = threading.Thread(target=audio_player_thread, args=(self.audio_jitter_buffer, self.running_flag))
        self.threads['feedback'] = threading.Thread(target=feedback_sender_thread, args=(self.sockets['control'], self.server_address, self.monitor, self.running_flag))
        for thread in self.threads.values():
            thread.daemon = True
            thread.start()
        display_thread(self)

    def cleanup(self):
        self.running_flag['running'] = False
        # No need to sleep, socket closure will interrupt threads
        for sock in self.sockets.values():
            if sock: sock.close()
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=0.5)
        self.threads.clear()
        self.sockets = {'video': None, 'audio': None, 'control': None}
        self.is_connected = False
        self.connect_btn.config(text="连接")
        self.play_btn.config(state=tk.DISABLED)
        self.video_listbox.delete(0, tk.END)
        self.status_label.config(text="状态: 未连接")
        placeholder_img = Image.new('RGB', (1, 1), color='black')
        placeholder_tk = ImageTk.PhotoImage(image=placeholder_img)
        self.video_label.config(image=placeholder_tk)
        self.video_label.image = placeholder_tk

    def on_closing(self):
        self.disconnect()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()