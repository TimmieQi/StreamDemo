# qt_client.py (最终优化版)

import sys
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
    QMenu, QSlider, QComboBox, QSizePolicy, QStyle
)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, Signal, QObject

from shared_config import *

# 用于在非UI线程和UI线程间通信的信号对象
class WorkerSignals(QObject):
    play_info_received = Signal(dict)
    play_failed = Signal(str)

class MasterClock:
    def __init__(self):
        self.reset()

    def reset(self):
        self._lock = threading.RLock()
        self._start_pts_ms = -1
        self._start_time = -1
        self._paused_at_time = -1
        self._rate = 1.0

    @property
    def is_paused(self):
        return self._paused_at_time != -1

    def start(self, pts_ms, audio_latency_sec=0.0):
        with self._lock:
            if self._start_time == -1 and pts_ms is not None:
                self._start_pts_ms = pts_ms
                # 关键修复：补偿音频输出延迟
                self._start_time = time.time() - audio_latency_sec
                print(f"[时钟] 主时钟启动。初始PTS: {pts_ms}ms, 音频延迟补偿: {audio_latency_sec*1000:.2f}ms")

    def get_time_ms(self):
        with self._lock:
            if self._start_time == -1:
                return -1
            if self.is_paused:
                elapsed_before_pause = self._paused_at_time - self._start_time
                return self._start_pts_ms + int(elapsed_before_pause * 1000 * self._rate)

            elapsed = time.time() - self._start_time
            return self._start_pts_ms + int(elapsed * 1000 * self._rate)

    def pause(self):
        with self._lock:
            if not self.is_paused and self._start_time != -1:
                self._paused_at_time = time.time()

    def resume(self):
        with self._lock:
            if self.is_paused:
                paused_duration = time.time() - self._paused_at_time
                self._start_time += paused_duration
                self._paused_at_time = -1

    def set_rate(self, rate):
        self._rate = 1.0

class NetworkMonitor:
    def __init__(self):
        self.reset()
    def reset(self):
        self.lock = threading.Lock()
        self.received_packets, self.lost_packets, self.expected_frame_id = 0, 0, -1
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

class VideoJitterBuffer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.lock = threading.Lock()
        self.packet_buffers = defaultdict(lambda: {"packets": {}, "total_packets": -1, "pts": -1})
        self.ready_queue = deque()
        self.last_played_pts = -1
    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        if len(packet) < 16: return
        frame_id, pts_ms, p_idx, total_p = int.from_bytes(packet[0:4], 'big'), int.from_bytes(packet[4:12], 'big'), int.from_bytes(packet[12:14], 'big'), int.from_bytes(packet[14:16], 'big')
        with self.lock:
            if self.last_played_pts != -1 and pts_ms < self.last_played_pts: return
            if p_idx in self.packet_buffers[frame_id]["packets"]: return
            if not self.packet_buffers[frame_id]["packets"]: monitor.record_packet(frame_id)
            buffer = self.packet_buffers[frame_id]
            buffer["packets"][p_idx], buffer["total_packets"], buffer["pts"] = packet[16:], total_p, pts_ms
            if len(buffer["packets"]) == buffer["total_packets"]: self._push_to_ready_queue(frame_id)
    def _push_to_ready_queue(self, frame_id):
        buffer = self.packet_buffers.pop(frame_id)
        data = b"".join(buffer["packets"][i] for i in sorted(buffer["packets"]))
        self.ready_queue.append((buffer["pts"], data))
        self.ready_queue = deque(sorted(self.ready_queue))
    def get_frame(self, target_pts_ms):
        with self.lock:
            if not self.ready_queue or target_pts_ms == -1: return None
            best_frame = None
            for i in range(len(self.ready_queue)):
                pts, data = self.ready_queue[i]
                if pts <= target_pts_ms: best_frame = (pts, data)
                else: break
            if best_frame:
                self.last_played_pts = best_frame[0]
                while self.ready_queue and self.ready_queue[0][0] <= self.last_played_pts:
                    self.ready_queue.popleft()
                return best_frame[1]
            return None

class AudioJitterBuffer:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.silence = b'\x00' * (AUDIO_CHUNK * 2 * AUDIO_CHANNELS)
        self.reset()
    def reset(self):
        self.lock = threading.Lock()
        self.buffer = []
        self.expected_seq = -1
    def add_chunk(self, chunk):
        if len(chunk) < 16: return
        seq, pts_ms, payload = int.from_bytes(chunk[0:8], 'big'), int.from_bytes(chunk[8:16], 'big'), chunk[16:]
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq
            if seq >= self.expected_seq and len(self.buffer) < self.max_size:
                heapq.heappush(self.buffer, (seq, pts_ms, payload))
    def get_chunk(self):
        with self.lock:
            if not self.buffer: return None, None
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
                return None, self.silence
    def clear(self):
        self.reset()

# --- 线程函数 ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag):
    while running_flag.get('running'):
        try:
            jitter_buffer.add_packet(sock.recvfrom(65535)[0], monitor)
        except (socket.timeout, socket.error):
            if not running_flag.get('running'): break
            continue

def audio_receiver_thread(sock, audio_buffer, running_flag):
    while running_flag.get('running'):
        try:
            audio_buffer.add_chunk(sock.recvfrom(16 + AUDIO_CHUNK * 2)[0])
        except (socket.timeout, socket.error):
            if not running_flag.get('running'): break
            continue

def audio_player_thread(audio_buffer, clock, state_vars, running_flag):
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, output=True)

        while running_flag.get('running'):
            is_file_stream = state_vars.get('duration_sec', 0) > 0
            if is_file_stream and clock.is_paused:
                if stream.is_active():
                    stream.stop_stream()
                time.sleep(0.01)
                continue

            if not stream.is_active():
                stream.start_stream()

            pts_ms, chunk = audio_buffer.get_chunk()
            if not chunk:
                time.sleep(0.005)
                continue

            # 关键修复：在第一次获取到数据时，用声卡延迟来启动时钟
            if clock.get_time_ms() == -1:
                latency = stream.get_output_latency()
                clock.start(pts_ms, latency)

            if state_vars['volume'] < 1.0:
                samples = np.frombuffer(chunk, dtype=np.int16)
                samples = (samples * state_vars['volume']).astype(np.int16)
                stream.write(samples.tobytes())
            else:
                stream.write(chunk)
    except Exception as e:
        import traceback
        print(f"[客户端-播放] [致命错误] 音频播放时发生异常: {e}")
        traceback.print_exc()
    finally:
        if stream:
            stream.close()
        p.terminate()
        print("[客户端-播放] 音频播放线程已停止。")

def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    while running_flag.get('running'):
        time.sleep(1)
        if not running_flag.get('running'): break
        try:
            stats = monitor.get_statistics()
            stats['command'] = 'heartbeat'
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error:
            break

# --- QT6主窗口类 ---
class VideoStreamClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.running_flag = {'running': False}
        self.sockets, self.threads = {}, {}
        self.server_address, self.current_source = None, "无"
        self.scale_mode, self.last_frame = "fit", None

        self.PLAYER_STATE_STOPPED, self.PLAYER_STATE_LOADING, self.PLAYER_STATE_PLAYING = 0, 1, 2
        self.player_status = self.PLAYER_STATE_STOPPED

        self.playback_state = { 'duration_sec': 0, 'volume': 1.0, 'rate': 1.0 }
        self.monitor = NetworkMonitor()
        self.video_jitter_buffer = VideoJitterBuffer()
        self.audio_jitter_buffer = AudioJitterBuffer()
        self.master_clock = MasterClock()

        self.worker_signals = WorkerSignals()
        self.worker_signals.play_info_received.connect(self.on_play_info_received)
        self.worker_signals.play_failed.connect(self.on_play_failed)

        self.init_ui()
        self.start_ui_updater()

    def init_ui(self):
        self.setWindowTitle("高级视频流客户端 (优化版)")
        self.setGeometry(100, 100, 1000, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        conn_group = QWidget()
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("服务器IP:"))
        self.ip_entry = QLineEdit("127.0.0.1")
        conn_layout.addWidget(self.ip_entry)
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)
        left_layout.addWidget(conn_group)
        self.video_list = QListWidget()
        left_layout.addWidget(QLabel("播放列表:"))
        left_layout.addWidget(self.video_list)
        self.play_btn = QPushButton("播放选中项")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_selected)
        left_layout.addWidget(self.play_btn)

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel("请连接服务器并选择一个视频源")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        right_layout.addWidget(self.video_label, stretch=1)

        # 控制条
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self.seek_slider_moved)
        self.progress_slider.sliderReleased.connect(self.seek_slider_released)
        controls_layout.addWidget(self.progress_slider)

        bottom_bar = QHBoxLayout()
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_pause)
        bottom_bar.addWidget(self.play_pause_btn)

        self.time_label = QLabel("00:00 / 00:00")
        bottom_bar.addWidget(self.time_label)
        bottom_bar.addStretch()

        bottom_bar.addWidget(QLabel("音量:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMaximumWidth(150)
        self.volume_slider.valueChanged.connect(self.change_volume)
        bottom_bar.addWidget(self.volume_slider)

        bottom_bar.addWidget(QLabel("速率:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1.0x"])
        self.speed_combo.setCurrentIndex(0)
        self.speed_combo.setEnabled(False)
        bottom_bar.addWidget(self.speed_combo)
        controls_layout.addLayout(bottom_bar)
        right_layout.addWidget(controls_widget)

        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=3)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("状态: 未连接")
        self.setStatusBar(self.status_bar)

        self.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_label.customContextMenuRequested.connect(self.show_context_menu)

    def start_ui_updater(self):
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui_tick)
        self.ui_timer.start(20) # 提高UI更新频率以获得更流畅的视频

    def update_ui_tick(self):
        if self.player_status == self.PLAYER_STATE_PLAYING:
            is_file_stream = self.playback_state['duration_sec'] > 0
            if not (is_file_stream and self.master_clock.is_paused):
                now_ms = self.master_clock.get_time_ms()
                if now_ms != -1:
                    frame_data = self.video_jitter_buffer.get_frame(now_ms)
                    if frame_data:
                        try:
                            nparr = np.frombuffer(frame_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                self.display_frame()
                        except Exception: pass

            if is_file_stream and now_ms != -1:
                current_sec = now_ms / 1000.0
                total_sec = self.playback_state['duration_sec']
                if not self.progress_slider.isSliderDown():
                    self.progress_slider.setValue(int(current_sec / total_sec * 1000))
                self.time_label.setText(f"{self.format_time(current_sec)} / {self.format_time(total_sec)}")

    def display_frame(self):
        if self.last_frame is None or self.player_status != self.PLAYER_STATE_PLAYING: return
        h, w, ch = self.last_frame.shape
        qImg = QImage(self.last_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        if self.scale_mode == "adapt": pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        elif self.scale_mode != "original": pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def format_time(self, seconds):
        s = int(seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

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
            for s in ['video', 'audio']: self.sockets[s].settimeout(1.0)
            self.sockets['video'].bind(('', VIDEO_PORT))
            self.sockets['audio'].bind(('', AUDIO_PORT))

            self.sockets['control'].settimeout(5)
            self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048)
            self.sockets['control'].settimeout(None)

            self.video_list.clear()
            for item in json.loads(data.decode()):
                self.video_list.addItem(item)

            self.start_threads()
            self.is_connected = True
            self.connect_btn.setText("断开")
            self.play_btn.setEnabled(True)
            self.status_bar.showMessage("状态: 连接成功，请选择播放项")
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
        self.player_status = self.PLAYER_STATE_STOPPED
        if not self.running_flag.get('running'): return

        self.running_flag['running'] = False
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=1.5)

        for sock in self.sockets.values():
            sock.close()

        self.threads.clear()
        self.sockets.clear()
        self.is_connected = False
        self.current_source = "无"
        self.last_frame = None

        self.connect_btn.setText("连接")
        self.play_btn.setEnabled(False)
        self.status_bar.showMessage("状态: 未连接")
        self.reset_playback_ui()

    def start_threads(self):
        thread_map = {
            'video_recv': (video_receiver_thread, (self.sockets['video'], self.video_jitter_buffer, self.monitor, self.running_flag)),
            'audio_recv': (audio_receiver_thread, (self.sockets['audio'], self.audio_jitter_buffer, self.running_flag)),
            'audio_play': (audio_player_thread, (self.audio_jitter_buffer, self.master_clock, self.playback_state, self.running_flag)),
            'feedback': (feedback_sender_thread, (self.sockets['control'], self.server_address, self.monitor, self.running_flag)),
        }
        for name, (target, args) in thread_map.items():
            self.threads[name] = threading.Thread(target=target, args=args, daemon=True)
            self.threads[name].start()

    def play_selected(self):
        selected_items = self.video_list.selectedItems()
        if not selected_items:
            self.status_bar.showMessage("提示: 请先选择一个视频")
            return

        self.current_source = selected_items[0].text()
        print(f"\n[客户端] 请求播放: {self.current_source}")

        self.reset_playback_state()
        self.player_status = self.PLAYER_STATE_LOADING
        self.video_label.setText("加载中...")
        self.status_bar.showMessage(f"状态: 正在请求播放 {self.current_source}...")

        threading.Thread(target=self.request_play_worker, daemon=True).start()

    def request_play_worker(self):
        try:
            self.sockets['control'].sendto(json.dumps({"command": "play", "source": self.current_source}).encode(), self.server_address)
            self.sockets['control'].settimeout(5.0)
            data, _ = self.sockets['control'].recvfrom(1024)
            self.sockets['control'].settimeout(None)
            response = json.loads(data.decode())
            if response.get("command") == "play_info":
                self.worker_signals.play_info_received.emit(response)
            else:
                self.worker_signals.play_failed.emit("错误: 服务器响应无效")
        except Exception as e:
            self.worker_signals.play_failed.emit(f"播放失败: {str(e)}")

    def on_play_info_received(self, response):
        self.player_status = self.PLAYER_STATE_PLAYING
        self.playback_state['duration_sec'] = response.get("duration", 0)
        self.setup_ui_for_playback(is_file_stream=(self.playback_state['duration_sec'] > 0))
        self.status_bar.showMessage(f"状态: 正在播放 {self.current_source}")

    def on_play_failed(self, error_message):
        self.player_status = self.PLAYER_STATE_STOPPED
        self.status_bar.showMessage(error_message)
        self.reset_playback_ui()

    def toggle_pause(self):
        if self.master_clock.is_paused:
            self.master_clock.resume()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.master_clock.pause()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def seek_slider_moved(self, value):
        if self.playback_state['duration_sec'] > 0:
            target_sec = self.playback_state['duration_sec'] * (value / 1000.0)
            self.time_label.setText(f"{self.format_time(target_sec)} / {self.format_time(self.playback_state['duration_sec'])}")

    def seek_slider_released(self):
        if self.playback_state['duration_sec'] == 0: return
        target_sec = self.playback_state['duration_sec'] * (self.progress_slider.value() / 1000.0)
        print(f"[客户端] 请求跳转到 {target_sec:.2f}s")
        try:
            self.sockets['control'].sendto(json.dumps({"command": "seek", "time": target_sec}).encode(), self.server_address)
            self.reset_playback_state()
            if self.master_clock.is_paused:
                self.toggle_pause()
        except Exception as e:
            self.status_bar.showMessage(f"跳转失败: {e}")

    def change_volume(self, value):
        self.playback_state['volume'] = value / 100.0

    def change_speed(self, index):
        self.playback_state['rate'] = 1.0
        self.master_clock.set_rate(1.0)

    def reset_playback_state(self):
        self.video_jitter_buffer.reset()
        self.audio_jitter_buffer.reset()
        self.master_clock.reset()

    def reset_playback_ui(self):
        self.video_label.clear()
        self.video_label.setText("请连接服务器并选择一个视频源")
        self.time_label.setText("00:00 / 00:00")
        self.progress_slider.setValue(0)
        self.progress_slider.setEnabled(False)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.speed_combo.setEnabled(False)

    def setup_ui_for_playback(self, is_file_stream):
        self.progress_slider.setEnabled(is_file_stream)
        self.play_pause_btn.setEnabled(is_file_stream)
        self.speed_combo.setEnabled(False) # 变速功能在此版本中永久禁用

        if is_file_stream:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.progress_slider.setRange(0, 1000)
        else:
            self.time_label.setText("直播")
            self.progress_slider.setValue(0)
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.play_pause_btn.setEnabled(False)

    def show_context_menu(self, pos):
        menu = QMenu(self)
        actions = {"自适应缩放 (Adapt)": "adapt", "按比例缩放 (Fit)": "fit", "原始大小 (Original)": "original"}
        for text, mode in actions.items():
            action = QAction(text, self, checkable=True, checked=(self.scale_mode == mode))
            action.triggered.connect(lambda checked, m=mode: self.set_scale_mode(m))
            menu.addAction(action)
        menu.exec(self.video_label.mapToGlobal(pos))

    def set_scale_mode(self, mode):
        self.scale_mode = mode
        if self.last_frame is not None:
            self.display_frame()

    def closeEvent(self, event):
        self.disconnect()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamClient()
    window.show()
    sys.exit(app.exec())