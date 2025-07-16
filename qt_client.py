# qt_client.py (H.265 + RTP over UDP 重构版)

import sys
import cv2
import socket
import numpy as np
import threading
import time
import json
from collections import deque
import heapq
import pyaudio
import av

# Matplotlib imports for charting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.font_manager as fm  # Import font_manager
from matplotlib import rcParams  # Import rcParams

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QLineEdit, QStatusBar, QMenu, QSlider, QComboBox, QSizePolicy, QStyle
)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRect, QEvent  # Import QRect and QEvent

from shared_config import *

# --- Matplotlib 中文字体配置 ---
# 尝试设置支持中文的字体
# 优先使用常见的Windows中文字体，如果不存在则尝试macOS字体，最后使用通用sans-serif
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Apple Color Emoji', 'Segoe UI Emoji',
                                   'Segoe UI Symbol', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# --- 信号 (无变化) ---
class WorkerSignals(QObject):
    play_info_received = Signal(dict)
    play_failed = Signal(str)


# --- [已重构] 主时钟 ---
class MasterClock:
    def __init__(self):
        self.reset()

    def reset(self):
        self._lock = threading.RLock()
        self._current_pts_ms = -1
        self._paused = False

    def start(self, pts_ms):
        """仅在第一次接收到音频时调用，以设置初始时间"""
        with self._lock:
            if self._current_pts_ms == -1 and pts_ms is not None:
                self._current_pts_ms = pts_ms
                print(f"[时钟] 主时钟启动。初始PTS: {pts_ms}ms")

    def update_time(self, pts_ms):
        """由音频播放线程调用，用实际播放的PTS来驱动时钟前进"""
        with self._lock:
            if not self._paused and pts_ms is not None:
                self._current_pts_ms = pts_ms

    def get_time_ms(self):
        """获取由音频驱动的当前播放时间"""
        with self._lock:
            return self._current_pts_ms

    @property
    def is_paused(self):
        return self._paused

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def set_rate(self, rate):
        pass


# --- 网络监控 (已修改以跟踪字节数) ---
class NetworkMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.received_packets = 0
        self.lost_packets = 0
        self.expected_seq = -1
        self.total_bytes_received = 0
        self.last_reset_time = time.time()

    def record_packet(self, seq, packet_size):
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq
            if seq > self.expected_seq: self.lost_packets += seq - self.expected_seq
            self.expected_seq = seq + 1
            self.received_packets += 1
            self.total_bytes_received += packet_size

    def get_statistics(self):
        with self.lock:
            total = self.received_packets + self.lost_packets
            loss_rate = self.lost_packets / total if total > 0 else 0.0

            current_time = time.time()
            time_diff = current_time - self.last_reset_time
            bitrate_bps = (self.total_bytes_received * 8) / time_diff if time_diff > 0 else 0

            stats = {
                "loss_rate": loss_rate,
                "bitrate_bps": bitrate_bps,
            }
            self.received_packets = 0
            self.lost_packets = 0
            self.total_bytes_received = 0
            self.last_reset_time = current_time
            return stats


# --- 视频Jitter Buffer ---
class VideoPacketJitterBuffer:
    def __init__(self):
        self.max_size = 200
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.buffer = []
        self.expected_seq = -1

    def add_packet(self, packet, monitor):
        if len(packet) < 6: return
        seq = int.from_bytes(packet[0:2], 'big')
        monitor.record_packet(seq, len(packet))
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq
            if seq >= self.expected_seq and len(self.buffer) < self.max_size:
                heapq.heappush(self.buffer, (seq, packet))

    def get_packet(self):
        with self.lock:
            if not self.buffer: return None
            seq, packet = self.buffer[0]
            if seq == self.expected_seq:
                heapq.heappop(self.buffer)
                self.expected_seq = (self.expected_seq + 1) % (2 ** 16)
                return packet
            elif seq < self.expected_seq:
                heapq.heappop(self.buffer)
                return self.get_packet()
            else:
                seq, packet = heapq.heappop(self.buffer)
                self.expected_seq = (seq + 1) % (2 ** 16)
                return packet


# --- 解码后帧的缓冲 ---
class DecodedFrameBuffer:
    def __init__(self):
        self.buffer_size_ms = 500
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.queue = deque()
        self.last_played_pts = -1

    def add_frame(self, frame_with_pts_and_image):
        with self.lock:
            self.queue.append(frame_with_pts_and_image)
            self.queue = deque(sorted(self.queue, key=lambda x: x[0]))

    def get_frame(self, target_pts_ms):
        with self.lock:
            if not self.queue or target_pts_ms == -1: return None, None
            best_frame_pts = None
            best_frame_img_data = None
            for pts, img_data in self.queue:
                if pts <= target_pts_ms:
                    best_frame_pts = pts
                    best_frame_img_data = img_data
                else:
                    break
            if best_frame_img_data is not None:
                self.last_played_pts = best_frame_pts
                while self.queue and self.queue[0][0] <= self.last_played_pts:
                    self.queue.popleft()
                return best_frame_pts, best_frame_img_data
            return None, None


# --- 音频Jitter Buffer ---
class AudioJitterBuffer:
    def __init__(self):
        self.max_size = 200
        self.silence = b'\x00' * (AUDIO_CHUNK * 2 * AUDIO_CHANNELS)
        self.reset()

    def reset(self):
        self.lock = threading.Lock();
        self.buffer = [];
        self.expected_seq = -1

    def add_chunk(self, chunk):
        if len(chunk) < 8: return
        seq = int.from_bytes(chunk[0:4], 'big')
        ts = int.from_bytes(chunk[4:8], 'big', signed=True)
        payload = chunk[8:]
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq
            if seq >= self.expected_seq and len(self.buffer) < self.max_size:
                heapq.heappush(self.buffer, (seq, ts, payload))

    def get_chunk(self):
        with self.lock:
            if not self.buffer: return None, None
            seq, ts, payload = self.buffer[0]
            if seq == self.expected_seq:
                heapq.heappop(self.buffer);
                self.expected_seq += 1
                return ts, payload
            elif seq < self.expected_seq:
                heapq.heappop(self.buffer);
                return self.get_chunk()
            else:
                seq, payload = heapq.heappop(self.buffer)
                self.expected_seq += 1;
                return None, self.silence  # Keep ts as None if packet is dropped

    def clear(self):
        self.reset()


# --- 线程函数 ---
def video_receiver_thread(sock, packet_buffer, monitor, running_flag):
    while running_flag.get('running'):
        try:
            packet_data = sock.recv(65535)
            packet_buffer.add_packet(packet_data, monitor)
        except (socket.timeout, socket.error):
            if not running_flag.get('running'): break
            continue


def audio_receiver_thread(sock, audio_buffer, running_flag):
    while running_flag.get('running'):
        try:
            audio_buffer.add_chunk(sock.recvfrom(8 + AUDIO_CHUNK * 2)[0])
        except (socket.timeout, socket.error):
            if not running_flag.get('running'): break
            continue


def video_decoder_thread(packet_buffer, frame_buffer, clock, running_flag):
    print("[客户端-解码] 视频解码线程启动。")
    try:
        codec = av.codec.Codec(VIDEO_CODEC, 'r')
        decoder = av.codec.context.CodecContext.create(VIDEO_CODEC, 'r')
    except Exception as e:
        print(f"[客户端-解码] [致命错误] 无法创建解码器: {e}")
        return

    # 用于缓存分片数据的字典 {timestamp: [payload_chunk1, payload_chunk2, ...]}
    reassembly_buffer = {}

    while running_flag.get('running'):
        packet_data = packet_buffer.get_packet()
        if packet_data is None:
            time.sleep(0.005)
            continue

        # 新协议解析: Header(7B) + Payload
        if len(packet_data) < 7:
            continue

        ts = int.from_bytes(packet_data[2:6], 'big', signed=True)
        frag_info = int.from_bytes(packet_data[6:7], 'big')
        payload = packet_data[7:]

        is_start = (frag_info & 0x80) != 0
        is_end = (frag_info & 0x40) != 0

        # ---- 帧重组逻辑 ----
        full_frame_payload = None

        if is_start and is_end:
            # 完整包，未分片
            full_frame_payload = payload
        else:
            # 分片包
            if is_start:
                # 新的一帧的开始，清空旧的（可能不完整的）数据并开始记录
                reassembly_buffer[ts] = [payload]
            elif ts in reassembly_buffer:
                # 中间或结尾的包，追加数据
                reassembly_buffer[ts].append(payload)

            if is_end and ts in reassembly_buffer:
                # 收到结尾，可以重组了
                fragments = reassembly_buffer.pop(ts)
                full_frame_payload = b''.join(fragments)
        # ---- 帧重组逻辑结束 ----

        if full_frame_payload:
            try:
                av_packet = av.Packet(full_frame_payload)
                av_packet.pts = ts
                for frame in decoder.decode(av_packet):
                    img = frame.to_ndarray(format='rgb24')
                    frame_buffer.add_frame((frame.pts, img))
            except Exception as e:
                pass

        # 清理过时的重组缓冲区条目，防止内存泄漏
        # (如果一个分片帧的结尾丢失，它会永远留在缓冲区里)
        # 简单策略：只保留最新的几个时间戳
        if len(reassembly_buffer) > 10:
            oldest_ts = min(reassembly_buffer.keys())
            del reassembly_buffer[oldest_ts]

    print("[客户端-解码] 视频解码线程已停止。")


# --- 音频播放线程 ---
def audio_player_thread(audio_buffer, clock, state_vars, running_flag):
    p = pyaudio.PyAudio();
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, output=True)
        while running_flag.get('running'):
            is_file_stream = state_vars.get('duration_sec', 0) > 0
            if is_file_stream and clock.is_paused:
                if stream.is_active(): stream.stop_stream()
                time.sleep(0.01);
                continue

            if not stream.is_active() and not clock.is_paused: stream.start_stream()

            pts_ms, chunk = audio_buffer.get_chunk()

            if not chunk:
                time.sleep(0.01)
                continue

            if clock.get_time_ms() == -1 and pts_ms is not None:
                clock.start(pts_ms)

            if pts_ms is not None:
                clock.update_time(pts_ms)

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
        if stream: stream.close()
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


# --- ChartWidget 类 ---
class ChartWidget(QWidget):
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.data_history = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.start_time = time.time()

        self.figure, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        self.ax.set_title(title, fontsize=14, color='white')
        self.ax.set_ylabel(y_label, fontsize=12, color='white')
        self.ax.tick_params(axis='x', labelsize=10, colors='white')
        self.ax.tick_params(axis='y', labelsize=10, colors='white')
        self.ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        self.ax.set_facecolor("#222222")
        self.figure.patch.set_facecolor("#222222")

        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)

        self.area_plot = None
        self.line = None
        self.figure.tight_layout(pad=0.5)

    def update_chart(self, value):
        current_time = time.time()

        if not self.timestamps:
            self.start_time = current_time

        relative_time = current_time - self.start_time

        self.data_history.append(value)
        self.timestamps.append(relative_time)

        if self.area_plot:
            self.area_plot.remove()
            self.area_plot = None
        if self.line:
            self.line.remove()
            self.line = None

        self.area_plot = self.ax.fill_between(
            list(self.timestamps),
            list(self.data_history),
            color='cyan',
            alpha=0.4,
            label=self.ax.get_ylabel()
        )
        self.line, = self.ax.plot(list(self.timestamps), list(self.data_history), color='cyan', linewidth=1)

        if self.timestamps:
            min_x = max(0, self.timestamps[-1] - 10)
            max_x = self.timestamps[-1] + 1
            self.ax.set_xlim(min_x, max_x)
        else:
            self.ax.set_xlim(0, 10)

        if self.data_history:
            min_y = min(self.data_history)
            max_y = max(self.data_history)
            y_range = max_y - min_y

            if y_range == 0:
                padding = 1.0
                min_y_lim = min_y - padding if min_y > 0 else 0
                max_y_lim = max_y + padding
            else:
                padding = y_range * 0.1
                min_y_lim = min_y - padding
                max_y_lim = max_y + padding
                if min_y >= 0:
                    min_y_lim = max(0, min_y_lim)

            self.ax.set_ylim(min_y_lim, max_y_lim)
        else:
            self.ax.set_ylim(0, 10)

        self.canvas.draw_idle()

    def clear_chart(self):
        self.data_history.clear()
        self.timestamps.clear()
        self.start_time = time.time()
        if self.area_plot:
            self.area_plot.remove()
            self.area_plot = None
        if self.line:
            self.line.remove()
            self.line = None
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.canvas.draw_idle()


# --- DebugWindow 类 ---
class DebugWindow(QMainWindow):
    def __init__(self, bitrate_chart, fps_chart, latency_chart, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高级调试 - 实时图表")
        self.setGeometry(150, 150, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.bitrate_chart = bitrate_chart
        self.fps_chart = fps_chart
        self.latency_chart = latency_chart

        layout.addWidget(self.bitrate_chart)
        layout.addWidget(self.fps_chart)
        layout.addWidget(self.latency_chart)

    def closeEvent(self, event):
        if self.parent():
            self.parent().debug_window = None
        event.accept()


# --- QT6主窗口类 ---
class VideoStreamClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.running_flag = {'running': False}
        self.sockets = {}
        self.server_address = None
        self.current_source = "无"
        self.scale_mode = "fit"
        self.last_frame = None
        self.PLAYER_STATE_STOPPED, self.PLAYER_STATE_LOADING, self.PLAYER_STATE_PLAYING = 0, 1, 2
        self.player_status = self.PLAYER_STATE_STOPPED
        self.playback_state = {'duration_sec': 0, 'volume': 1.0, 'rate': 1.0, 'playback_finished': False}

        self.monitor = NetworkMonitor()
        self.video_packet_jitter_buffer = VideoPacketJitterBuffer()
        self.decoded_frame_buffer = DecodedFrameBuffer()
        self.audio_jitter_buffer = AudioJitterBuffer()
        self.master_clock = MasterClock()

        self.worker_signals = WorkerSignals()
        self.worker_signals.play_info_received.connect(self.on_play_info_received)
        self.worker_signals.play_failed.connect(self.on_play_failed)

        self.frame_count = 0
        self.last_fps_update_time = time.time()
        self.current_fps = 0

        self.current_latency_ms = 0
        self.current_bitrate_kbps = 0

        self.bitrate_chart = ChartWidget("码率 (kbps)", "kbps")
        self.fps_chart = ChartWidget("帧率 (FPS)", "FPS")
        self.latency_chart = ChartWidget("时延 (ms)", "ms")

        self.debug_window = None

        self.main_layout = None
        self.left_panel_widget = None
        self.video_player_container = None
        self.controls_widget = None
        self.is_video_fullscreen = False
        self.original_geometry = None

        # Flag to differentiate single and double clicks
        self.double_click_pending = False
        self.single_click_timer = QTimer(self)
        self.single_click_timer.setSingleShot(True)
        self.single_click_timer.setInterval(QApplication.doubleClickInterval())
        self.single_click_timer.timeout.connect(self._handle_single_click_action)

        self.threads = {}

        self.init_ui()
        self.start_ui_updater()

    def init_ui(self):
        self.setWindowTitle("高级视频流客户端 (H.265版)")
        self.setGeometry(100, 100, 1000, 800)
        self.original_geometry = self.geometry()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QHBoxLayout(main_widget)

        self.left_panel_widget = QWidget()
        left_layout = QVBoxLayout(self.left_panel_widget)
        conn_group = QWidget()
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("服务器IP:"))
        self.ip_entry = QLineEdit("127.0.0.1")
        conn_layout.addWidget(self.ip_entry)
        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn)
        left_layout.addWidget(conn_group)

        # Connect double-click on video list to play selected item
        self.video_list = QListWidget()
        left_layout.addWidget(QLabel("播放列表:"))
        left_layout.addWidget(self.video_list)
        self.video_list.itemDoubleClicked.connect(self.play_selected)  # Connect double-click to play

        self.play_btn = QPushButton("播放选中项")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_selected)
        left_layout.addWidget(self.play_btn)

        self.debug_btn = QPushButton("高级调试 (图表)")
        self.debug_btn.clicked.connect(self.show_debug_window)
        left_layout.addWidget(self.debug_btn)

        self.video_player_container = QWidget()
        self.video_player_container_layout = QVBoxLayout(self.video_player_container)
        self.video_player_container_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("请连接服务器并选择一个视频源")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.video_label.mousePressEvent = self.video_label_clicked

        self.video_player_container_layout.addWidget(self.video_label, stretch=1)

        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self.seek_slider_moved)
        self.progress_slider.sliderReleased.connect(self.seek_slider_released)
        controls_layout.addWidget(self.progress_slider)

        bottom_bar = QHBoxLayout()
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setCheckable(True)  # Make it checkable
        self.play_pause_btn.setChecked(False)  # Initial state: not playing, so show play icon
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

        self.fullscreen_btn = QPushButton()
        try:
            self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaFullscreen))
        except AttributeError:
            try:
                self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaMaximize))
            except AttributeError:
                self.fullscreen_btn.setText("全屏")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        bottom_bar.addWidget(self.fullscreen_btn)

        controls_layout.addLayout(bottom_bar)
        self.video_player_container_layout.addWidget(self.controls_widget)

        self.main_layout.addWidget(self.left_panel_widget, stretch=1)
        self.main_layout.addWidget(self.video_player_container, stretch=3)

        self.status_bar = QStatusBar()
        self.status_bar.showMessage("状态: 未连接")
        self.setStatusBar(self.status_bar)
        self.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_label.customContextMenuRequested.connect(self.show_context_menu)

    def _handle_single_click_action(self):
        """This method is called by the timer if no double-click occurs."""
        # Only execute pause if a double-click was NOT detected
        if not self.double_click_pending:
            if self.player_status == self.PLAYER_STATE_PLAYING:
                self.toggle_pause()
        # Reset the flag after handling the single click (or after timer expires)
        self.double_click_pending = False

    def video_label_clicked(self, event):
        """Handle click on video label to toggle pause or fullscreen."""
        if event.button() == Qt.MouseButton.LeftButton:
            if event.type() == QEvent.MouseButtonDblClick:
                # If it's a double-click, set the flag and stop any pending single-click timer
                self.double_click_pending = True
                if self.single_click_timer.isActive():
                    self.single_click_timer.stop()
                self.toggle_fullscreen()
            elif event.type() == QEvent.MouseButtonPress:
                # If it's a single mouse press, start the timer for potential double-click.
                # The _handle_single_click_action will check the double_click_pending flag.
                self.single_click_timer.start()

    def format_time(self, seconds):
        s = int(seconds);
        return f"{s // 60:02d}:{s % 60:02d}"

    def connect(self):
        server_ip = self.ip_entry.text()
        if not server_ip: self.status_bar.showMessage("错误: 请输入服务器IP地址"); return
        self.server_address = (server_ip, CONTROL_PORT);
        self.running_flag['running'] = True
        try:
            self.sockets = {'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                            'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
                            'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)}
            for s in ['video', 'audio']: self.sockets[s].settimeout(1.0)
            self.sockets['video'].bind(('', VIDEO_PORT));
            self.sockets['audio'].bind(('', AUDIO_PORT))
            self.sockets['control'].settimeout(5);
            self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048);
            self.sockets['control'].settimeout(None)
            self.video_list.clear();
            [self.video_list.addItem(item) for item in json.loads(data.decode())]
            self.start_threads();
            self.is_connected = True;
            self.connect_btn.setText("断开");
            self.play_btn.setEnabled(True)
            self.status_bar.showMessage("状态: 连接成功，请选择播放项")
        except Exception as e:
            self.status_bar.showMessage(f"连接失败: {str(e)}");
            self.cleanup()

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def disconnect(self):
        self.player_status = self.PLAYER_STATE_STOPPED
        if not self.running_flag.get('running'): return
        self.running_flag['running'] = False

        for thread in self.threads.values():
            if thread.is_alive(): thread.join(timeout=1.0)
        for sock in self.sockets.values(): sock.close()
        self.threads.clear();
        self.sockets.clear();
        self.is_connected = False

        self.current_source = "无";
        self.last_frame = None

        self.reset_playback_state()

        if self.connect_btn and self.connect_btn.parent():
            self.connect_btn.setText("连接")
        if self.play_btn and self.play_btn.parent():
            self.play_btn.setEnabled(False)
        if self.status_bar and self.status_bar.parent():
            self.status_bar.showMessage("状态: 未连接")

        self.reset_playback_ui()

        if self.debug_window:
            self.debug_window.close()
            self.debug_window = None

    def start_ui_updater(self):
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui_tick)
        self.ui_timer.start(30)

    def update_ui_tick(self):
        now_ms = self.master_clock.get_time_ms()

        current_time = time.time()
        time_diff = current_time - self.monitor.last_reset_time
        if time_diff > 0:
            self.current_bitrate_kbps = (self.monitor.total_bytes_received * 8 / 1000) / time_diff

        self.bitrate_chart.update_chart(self.current_bitrate_kbps)

        if self.player_status != self.PLAYER_STATE_PLAYING:
            self.bitrate_chart.clear_chart()
            self.fps_chart.clear_chart()
            self.latency_chart.clear_chart()
            return

        if not self.master_clock.is_paused:
            if now_ms != -1:
                frame_pts, frame_data = self.decoded_frame_buffer.get_frame(now_ms)
                if frame_data is not None:
                    self.last_frame = frame_data
                    self.display_frame()
                    self.frame_count += 1

                    if frame_pts is not None:
                        self.current_latency_ms = max(0, now_ms - frame_pts)

                else:
                    self.current_latency_ms = 0
            else:
                self.current_latency_ms = 0

        if current_time - self.last_fps_update_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update_time)
            self.frame_count = 0
            self.last_fps_update_time = current_time

        self.fps_chart.update_chart(self.current_fps)
        self.latency_chart.update_chart(self.current_latency_ms)

        is_file_stream = self.playback_state['duration_sec'] > 0
        if is_file_stream:
            if self.playback_state.get('playback_finished', False):
                return

            total_sec = self.playback_state['duration_sec']
            if now_ms != -1:
                current_sec = now_ms / 1000.0

                if current_sec >= total_sec:
                    current_sec = total_sec
                    self.playback_state['playback_finished'] = True
                    self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                    print("[客户端] 视频播放结束。")
                    self.reset_playback_state()

                if not self.progress_slider.isSliderDown():
                    if total_sec > 0:
                        self.progress_slider.setValue(int(current_sec / total_sec * 1000))

                self.time_label.setText(f"{self.format_time(current_sec)} / {self.format_time(total_sec)}")

    def display_frame(self):
        if self.last_frame is None or self.player_status != self.PLAYER_STATE_PLAYING: return
        h, w, ch = self.last_frame.shape
        qImg = QImage(self.last_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Always use KeepAspectRatio for 'fit' and 'adapt' modes to prevent stretching
        if self.scale_mode == "adapt" or self.scale_mode == "fit":
            pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        # For "original" mode, the pixmap is not scaled by this block,
        # so it inherently maintains its original size and aspect ratio.

        self.video_label.setPixmap(pixmap)

    def start_threads(self):
        thread_map = {
            'video_recv': (video_receiver_thread,
                           (self.sockets['video'], self.video_packet_jitter_buffer, self.monitor, self.running_flag)),
            'video_decode': (video_decoder_thread,
                             (self.video_packet_jitter_buffer, self.decoded_frame_buffer, self.master_clock,
                              self.running_flag)),
            'audio_recv': (audio_receiver_thread, (self.sockets['audio'], self.audio_jitter_buffer, self.running_flag)),
            'audio_play': (audio_player_thread,
                           (self.audio_jitter_buffer, self.master_clock, self.playback_state, self.running_flag)),
            'feedback': (feedback_sender_thread,
                         (self.sockets['control'], self.server_address, self.monitor, self.running_flag))
        }
        self.threads.clear()
        for name, (target, args) in thread_map.items():
            thread = threading.Thread(target=target, args=args, daemon=True)
            self.threads[name] = thread
            thread.start()

    def reset_playback_state(self):
        print("[客户端] 重置所有播放状态和缓冲区...")
        self.video_packet_jitter_buffer.reset()
        self.decoded_frame_buffer.reset()
        self.audio_jitter_buffer.clear()
        self.master_clock.reset()
        self.monitor.reset()
        self.last_frame = None
        self.playback_state['playback_finished'] = False
        self.frame_count = 0
        self.last_fps_update_time = time.time()
        self.current_fps = 0
        self.current_latency_ms = 0
        self.current_bitrate_kbps = 0
        self.bitrate_chart.clear_chart()
        self.fps_chart.clear_chart()
        self.latency_chart.clear_chart()

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
            self.sockets['control'].sendto(json.dumps({"command": "play", "source": self.current_source}).encode(),
                                           self.server_address)
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
        self.playback_state['playback_finished'] = False
        self.setup_ui_for_playback(is_file_stream=(self.playback_state['duration_sec'] > 0))
        self.status_bar.showMessage(f"状态: 正在播放 {self.current_source}")
        # When playback starts, set button to 'playing' state (pause icon, checked)
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.play_pause_btn.setChecked(True)

    def on_play_failed(self, error_message):
        self.player_status = self.PLAYER_STATE_STOPPED
        self.status_bar.showMessage(error_message)
        self.reset_playback_ui()

    def toggle_pause(self, checked=False):
        if self.playback_state.get('playback_finished', False): return
        if self.player_status != self.PLAYER_STATE_PLAYING: return  # Only toggle pause if playing
        if self.master_clock.is_paused:
            self.master_clock.resume()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.play_pause_btn.setChecked(True)  # Set to checked state (pause icon)
        else:
            self.master_clock.pause()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_pause_btn.setChecked(False)  # Set to unchecked state (play icon)

    def seek_slider_moved(self, value):
        if self.playback_state['duration_sec'] > 0:
            target_sec = self.playback_state['duration_sec'] * (value / 1000.0)
            self.time_label.setText(
                f"{self.format_time(target_sec)} / {self.format_time(self.playback_state['duration_sec'])}")

    def seek_slider_released(self):
        if self.playback_state['duration_sec'] == 0: return
        target_sec = self.playback_state['duration_sec'] * (self.progress_slider.value() / 1000.0)
        print(f"[客户端] 请求跳转到 {target_sec:.2f}s")
        self.reset_playback_state()
        try:
            self.sockets['control'].sendto(json.dumps({"command": "seek", "time": target_sec}).encode(),
                                           self.server_address)
            if self.master_clock.is_paused: self.toggle_pause()
        except Exception as e:
            self.status_bar.showMessage(f"跳转失败: {e}")

    def change_volume(self, value):
        self.playback_state['volume'] = value / 100.0

    def change_speed(self, index):
        pass

    def reset_playback_ui(self):
        if self.video_label and self.video_label.parent():
            self.video_label.clear()
            self.video_label.setText("请连接服务器并选择一个视频源")
        if self.time_label and self.time_label.parent():
            self.time_label.setText("00:00 / 00:00")
        if self.progress_slider and self.progress_slider.parent():
            self.progress_slider.setValue(0)
            self.progress_slider.setEnabled(False)
        if self.play_pause_btn and self.play_pause_btn.parent():
            self.play_pause_btn.setEnabled(False)
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_pause_btn.setChecked(False)  # Ensure it's unchecked when reset
        if self.speed_combo and self.speed_combo.parent():
            self.speed_combo.setEnabled(False)

        self.bitrate_chart.clear_chart()
        self.fps_chart.clear_chart()
        self.latency_chart.clear_chart()

    def setup_ui_for_playback(self, is_file_stream):
        if self.progress_slider and self.progress_slider.parent():
            self.progress_slider.setEnabled(is_file_stream)
        if self.play_pause_btn and self.play_pause_btn.parent():
            self.play_pause_btn.setEnabled(is_file_stream)
        if self.speed_combo and self.speed_combo.parent():
            self.speed_combo.setEnabled(False)

        if is_file_stream:
            if self.play_pause_btn and self.play_pause_btn.parent():
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            if self.progress_slider and self.progress_slider.parent():
                self.progress_slider.setRange(0, 1000)
        else:
            if self.time_label and self.time_label.parent():
                self.time_label.setText("直播")
            if self.progress_slider and self.progress_slider.parent():
                self.progress_slider.setValue(0)
            if self.play_pause_btn and self.play_pause_btn.parent():
                self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
                self.play_pause_btn.setEnabled(False)

    def show_context_menu(self, pos):
        menu = QMenu(self);
        # Updated context menu options for clarity
        actions = {"自适应 (保持比例)": "adapt", "按比例缩放 (Fit)": "fit", "原始大小 (Original)": "original"}
        for text, mode in actions.items():
            action = QAction(text, self, checkable=True, checked=(self.scale_mode == mode))
            action.triggered.connect(lambda checked, m=mode: self.set_scale_mode(m));
            menu.addAction(action)
        menu.exec(self.video_label.mapToGlobal(pos))

    def set_scale_mode(self, mode):
        self.scale_mode = mode;
        self.display_frame()

    def show_debug_window(self):
        if self.debug_window is None:
            self.debug_window = DebugWindow(self.bitrate_chart, self.fps_chart, self.latency_chart, parent=self)
            self.debug_window.show()
        else:
            self.debug_window.activateWindow()

    def toggle_fullscreen(self):
        if self.is_video_fullscreen:
            # Exit fullscreen
            self.showNormal()
            # Restore to original geometry, and slightly reduce size
            original_width = self.original_geometry.width()
            original_height = self.original_geometry.height()
            reduced_width = max(200, original_width - 50)
            reduced_height = max(150, original_height - 50)
            self.setGeometry(self.original_geometry.x(), self.original_geometry.y(), reduced_width, reduced_height)

            # Re-show hidden widgets
            self.left_panel_widget.show()
            self.statusBar().show()
            # self.menuBar().show() # Uncomment if you have a menu bar

            # Restore main layout stretch factors
            self.main_layout.setStretchFactor(self.left_panel_widget, 1)
            self.main_layout.setStretchFactor(self.video_player_container, 3)

            # Restore fullscreen button icon
            try:
                self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaFullscreen))
            except AttributeError:
                try:
                    self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaMaximize))
                except AttributeError:
                    self.fullscreen_btn.setText("全屏")

            self.is_video_fullscreen = False
        else:
            # Enter fullscreen
            # Hide other widgets
            self.left_panel_widget.hide()
            self.statusBar().hide()
            # self.menuBar().hide() # Uncomment if you have a menu bar

            # Adjust main layout to make video_player_container take all space
            self.main_layout.setStretchFactor(self.left_panel_widget, 0)
            self.main_layout.setStretchFactor(self.video_player_container, 100)

            self.showFullScreen()

            # Change fullscreen button icon to exit fullscreen
            try:
                self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaRestoreDown))
            except AttributeError:
                try:
                    self.fullscreen_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaMinimize))
                except AttributeError:
                    self.fullscreen_btn.setText("退出全屏")

            self.is_video_fullscreen = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.is_video_fullscreen:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.disconnect()
        if self.debug_window:
            self.debug_window.close()
        event.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamClient()
    window.show()
    sys.exit(app.exec())