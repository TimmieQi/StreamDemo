# qt_client.py

import sys
# sys.path.append('d:/pythonWorkSpace/Vediostream') # 根据你的实际路径调整
import cv2
import socket
import numpy as np
import threading
import time
import json
from collections import defaultdict, deque
import heapq
import pyaudio

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QListWidget, QPushButton, QLineEdit, QStatusBar,
    QMenu
)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer

from shared_config import *

# ### 新增：主时钟类 ###
class MasterClock:
    """一个以音频播放为基准的主时钟"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._lock = threading.Lock()
        self._start_pts_ms = -1
        self._start_time = -1

    def start(self, pts_ms):
        """用第一个音频包的PTS启动时钟"""
        with self._lock:
            if self._start_time == -1 and pts_ms is not None:
                self._start_pts_ms = pts_ms
                self._start_time = time.time()
                print(f"[时钟] 主时钟已启动，初始PTS: {pts_ms}ms")

    def get_time_ms(self):
        """获取当前播放的PTS时间"""
        with self._lock:
            if self._start_time == -1:
                return -1
            elapsed = time.time() - self._start_time
            return self._start_pts_ms + int(elapsed * 1000)

# --- 视频辅助类 (修改以支持PTS) ---
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
            return {"loss_rate": loss_rate}
    def reset(self): self.lock = threading.Lock(); self.received_packets, self.lost_packets, self.expected_frame_id = 0, 0, -1

class VideoJitterBuffer:
    def __init__(self):
        self.reset()

    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        # 新头部: 帧ID(4) + PTS(8) + 包序号(2) + 总包数(2) = 16字节
        if len(packet) < 16: return
        frame_id = int.from_bytes(packet[0:4], 'big')
        pts_ms = int.from_bytes(packet[4:12], 'big')
        packet_index = int.from_bytes(packet[12:14], 'big')
        total_packets = int.from_bytes(packet[14:16], 'big')

        with self.lock:
            # 简单丢弃过旧的帧
            if self.last_played_pts != -1 and pts_ms < self.last_played_pts:
                return

            if packet_index in self.packet_buffers[frame_id]["packets"]: return
            if not self.packet_buffers[frame_id]["packets"]: monitor.record_packet(frame_id)

            buffer = self.packet_buffers[frame_id]
            buffer["packets"][packet_index] = packet[16:]
            buffer["total_packets"] = total_packets
            buffer["pts"] = pts_ms # 存储该帧的PTS

            if len(buffer["packets"]) == buffer["total_packets"]:
                self._push_to_ready_queue(frame_id)

    def _push_to_ready_queue(self, frame_id):
        buffer = self.packet_buffers.pop(frame_id)
        data = b"".join(buffer["packets"][i] for i in sorted(buffer["packets"]))
        # 队列中存储 (PTS, 数据)
        self.ready_queue.append((buffer["pts"], data))
        # 按PTS排序，确保播放顺序正确
        self.ready_queue = deque(sorted(self.ready_queue))

    def get_frame(self, target_pts_ms):
        """根据主时钟时间获取最合适的帧"""
        with self.lock:
            if not self.ready_queue or target_pts_ms == -1:
                return None

            best_frame = None
            # 寻找时间戳 <= 目标时间戳的最新一帧
            # 由于队列已排序，从头开始找即可
            for i in range(len(self.ready_queue)):
                pts, data = self.ready_queue[i]
                if pts <= target_pts_ms:
                    best_frame = (pts, data)
                else:
                    # 后面的帧都太新了，中断查找
                    break

            if best_frame:
                # 找到了合适的帧，丢弃所有比它旧的帧
                self.last_played_pts = best_frame[0]
                while self.ready_queue and self.ready_queue[0][0] <= self.last_played_pts:
                    self.ready_queue.popleft()
                return best_frame[1]

            return None

    def reset(self):
        self.lock = threading.Lock()
        self.packet_buffers = defaultdict(lambda: {"packets": {}, "total_packets": -1, "pts": -1})
        self.ready_queue = deque()
        self.last_played_pts = -1

# --- 音频抖动缓冲 (修改以支持PTS) ---
class AudioJitterBuffer:
    def __init__(self, max_size=100, chunk_size=AUDIO_CHUNK, sample_width=2):
        self.max_size = max_size
        self.silence = b'\x00' * (chunk_size * sample_width * AUDIO_CHANNELS)
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.buffer = [] # 使用heapq作为优先级队列
        self.expected_seq = -1

    def add_chunk(self, chunk):
        # 新头部: 序列号(8字节) + PTS(8字节) = 16字节
        if len(chunk) < 16: return
        seq = int.from_bytes(chunk[0:8], 'big')
        pts_ms = int.from_bytes(chunk[8:16], 'big')
        payload = chunk[16:]
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq

            if seq >= self.expected_seq and len(self.buffer) < self.max_size:
                # 存储 (序列号, PTS, 数据)
                heapq.heappush(self.buffer, (seq, pts_ms, payload))

    def get_chunk(self):
        with self.lock:
            if not self.buffer: return None

            seq, pts_ms, payload = self.buffer[0]
            if seq == self.expected_seq:
                heapq.heappop(self.buffer)
                self.expected_seq += 1
                return pts_ms, payload
            elif seq < self.expected_seq:
                heapq.heappop(self.buffer)
                return self.get_chunk()
            else:
                self.expected_seq += 1
                # 补充静音时，PTS为None
                return None, self.silence

    def clear(self):
        self.reset()

# --- 线程函数 (修改以支持同步) ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag):
    while running_flag.get('running'):
        try:
            data, _ = sock.recvfrom(65535)
            jitter_buffer.add_packet(data, monitor)
        except socket.timeout:
            continue
        except socket.error as e:
            if running_flag.get('running'): print(f"[客户端-视频接收] [错误] Socket异常: {e}")
            break

def audio_receiver_thread(sock, audio_buffer, running_flag):
    while running_flag.get('running'):
        try:
            # 16字节头 + 音频数据
            data, _ = sock.recvfrom(16 + AUDIO_CHUNK * 2)
            audio_buffer.add_chunk(data)
        except socket.timeout:
            continue
        except socket.error as e:
            if running_flag.get('running'): print(f"[客户端-接收] [错误] 音频接收线程遇到socket错误: {e}")
            break
    print("[客户端-接收] 音频接收线程已停止。")


def audio_player_thread(audio_buffer, clock, running_flag):
    print("[客户端-播放] 音频播放线程初始化...")
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=AUDIO_FORMAT,
                        channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,
                        output=True,
                        frames_per_buffer=AUDIO_CHUNK)
        print(f"[客户端-播放] PyAudio流已成功打开。")

        while running_flag.get('running'):
            pts_and_chunk = audio_buffer.get_chunk()
            if pts_and_chunk:
                pts_ms, chunk = pts_and_chunk
                # 使用第一个有效PTS启动主时钟
                clock.start(pts_ms)
                stream.write(chunk)
            else:
                time.sleep(0.005)
    except Exception as e:
        print(f"[客户端-播放] [致命错误] 音频播放时发生异常: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("[客户端-播放] 音频播放线程已停止。")

def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    while running_flag.get('running'):
        time.sleep(1)
        if not running_flag.get('running'): break
        try:
            stats = monitor.get_statistics(); stats['command'] = 'heartbeat'
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error: break

# --- QT6主窗口类 ---
class VideoStreamClient(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_connected = False
        self.running_flag = {'running': False}
        self.sockets = {}
        self.threads = {}
        self.server_address = None
        self.current_source = "无"
        self.scale_mode = "fit"
        self.last_frame = None

        self.monitor = NetworkMonitor()
        self.video_jitter_buffer = VideoJitterBuffer()
        self.audio_jitter_buffer = AudioJitterBuffer()
        self.master_clock = MasterClock() # 实例化主时钟

        self.init_ui()
        self.start_display_updater()

    def init_ui(self):
        self.setWindowTitle("实时视频流客户端(QT6)")
        self.setGeometry(100, 100, 1000, 700)
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        control_panel = QWidget(); control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        conn_group = QWidget(); conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("服务器IP:")); self.ip_entry = QLineEdit("127.0.0.1")
        conn_layout.addWidget(self.ip_entry); self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection); conn_layout.addWidget(self.connect_btn)
        control_layout.addWidget(conn_group)
        self.video_list = QListWidget(); control_layout.addWidget(QLabel("播放列表:"))
        control_layout.addWidget(self.video_list); self.play_btn = QPushButton("播放选中项")
        self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.play_selected)
        control_layout.addWidget(self.play_btn)
        self.video_label = QLabel(); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(control_panel, stretch=1); main_layout.addWidget(self.video_label, stretch=3)
        self.status_bar = QStatusBar(); self.status_bar.showMessage("状态: 未连接")
        self.setStatusBar(self.status_bar)
        self.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_label.customContextMenuRequested.connect(self.show_context_menu)

    def start_display_updater(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(20) # 定时器频率可以稍慢，约50fps，同步逻辑会选择正确的帧

    def update_video_frame(self):
        # ### 修改：视频同步逻辑 ###
        now_ms = self.master_clock.get_time_ms()
        if now_ms == -1: return # 时钟未启动

        frame_data = self.video_jitter_buffer.get_frame(now_ms)
        if not frame_data: return

        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame()
        except Exception as e:
            print(f"[客户端-显示] 解码错误: {e}")

    def display_frame(self):
        if self.last_frame is None: return
        height, width, channel = self.last_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.last_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        if self.scale_mode == "original": pixmap = QPixmap.fromImage(q_img)
        elif self.scale_mode == "adapt": pixmap = QPixmap.fromImage(q_img.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else: pixmap = QPixmap.fromImage(q_img.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.video_label.setPixmap(pixmap)

    def toggle_connection(self):
        if self.is_connected: self.disconnect()
        else: self.connect()

    def connect(self):
        server_ip = self.ip_entry.text()
        if not server_ip:
            self.status_bar.showMessage("错误: 请输入服务器IP地址")
            return
        self.server_address = (server_ip, CONTROL_PORT)
        self.running_flag['running'] = True
        try:
            self.sockets['control'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['video'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['audio'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sockets['video'].settimeout(1.0); self.sockets['audio'].settimeout(1.0)
            self.sockets['video'].bind(('', VIDEO_PORT)); self.sockets['audio'].bind(('', AUDIO_PORT))
            self.sockets['control'].settimeout(5)
            self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048); self.sockets['control'].settimeout(None)
            self.video_list.clear()
            for item in json.loads(data.decode()): self.video_list.addItem(item)
            self.start_threads()
            self.is_connected = True; self.connect_btn.setText("断开")
            self.play_btn.setEnabled(True); self.status_bar.showMessage("状态: 连接成功，请选择播放项")
        except Exception as e:
            self.status_bar.showMessage(f"连接失败: {str(e)}")
            self.cleanup()

    def disconnect(self):
        if self.sockets.get('control') and self.server_address:
            try:
                self.sockets['control'].sendto(json.dumps({"command": "stop"}).encode(), self.server_address)
            except socket.error: pass
        self.cleanup()

    def cleanup(self):
        if not self.running_flag.get('running'): return
        self.running_flag['running'] = False
        for name, thread in self.threads.items():
            if thread.is_alive(): thread.join(timeout=1.5)
        for name, sock in self.sockets.items(): sock.close()
        self.threads.clear(); self.sockets.clear(); self.is_connected = False
        self.current_source = "无"; self.last_frame = None
        self.connect_btn.setText("连接"); self.play_btn.setEnabled(False)
        self.video_list.clear(); self.status_bar.showMessage("状态: 未连接")
        self.video_label.clear(); self.video_jitter_buffer.reset()
        self.audio_jitter_buffer.clear(); self.monitor.reset(); self.master_clock.reset()

    def start_threads(self):
        thread_map = {
            'video_recv': (video_receiver_thread, (self.sockets['video'], self.video_jitter_buffer, self.monitor, self.running_flag)),
            'audio_recv': (audio_receiver_thread, (self.sockets['audio'], self.audio_jitter_buffer, self.running_flag)),
            'audio_play': (audio_player_thread, (self.audio_jitter_buffer, self.master_clock, self.running_flag)),
            'feedback': (feedback_sender_thread, (self.sockets['control'], self.server_address, self.monitor, self.running_flag)),
        }
        for name, (target, args) in thread_map.items():
            self.threads[name] = threading.Thread(target=target, args=args, daemon=True)
            self.threads[name].start()

    def play_selected(self):
        if not (selected_items := self.video_list.selectedItems()):
            self.status_bar.showMessage("提示: 请先选择一个视频")
            return
        self.current_source = selected_items[0].text()
        print(f"\n[客户端] 请求播放: {self.current_source}")
        self.video_jitter_buffer.reset()
        self.audio_jitter_buffer.clear()
        self.monitor.reset()
        self.master_clock.reset() # 重置主时钟
        try:
            self.sockets['control'].sendto(json.dumps({"command": "play", "source": self.current_source}).encode(), self.server_address)
            self.status_bar.showMessage(f"状态: 正在播放 {self.current_source}")
        except Exception as e:
            self.status_bar.showMessage(f"播放失败: {str(e)}")

    def show_context_menu(self, pos):
        menu = QMenu(self)
        adapt_action = QAction("自适应缩放 (Adapt)", self); adapt_action.triggered.connect(lambda: self.set_scale_mode("adapt"))
        fit_action = QAction("按比例缩放 (Fit)", self); fit_action.triggered.connect(lambda: self.set_scale_mode("fit"))
        original_action = QAction("原始大小 (Original)", self); original_action.triggered.connect(lambda: self.set_scale_mode("original"))
        for action, mode in [(adapt_action, "adapt"), (fit_action, "fit"), (original_action, "original")]:
            action.setCheckable(True); action.setChecked(self.scale_mode == mode)
        menu.addAction(adapt_action); menu.addAction(fit_action); menu.addAction(original_action)
        menu.exec(self.video_label.mapToGlobal(pos))

    def set_scale_mode(self, mode):
        self.scale_mode = mode
        if self.last_frame is not None: self.display_frame()

    def closeEvent(self, event):
        self.disconnect()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamClient()
    window.show()
    sys.exit(app.exec())