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

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QLineEdit, QStatusBar, QMenu, QSlider, QComboBox, QSizePolicy, QStyle
)
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QTimer, Signal, QObject

from shared_config import *

# --- 信号 (无变化) ---
class WorkerSignals(QObject):
    play_info_received = Signal(dict)
    play_failed = Signal(str)

# --- [已重构] 主时钟 ---
# 这是一个由音频播放驱动的状态报告时钟，不再基于系统时间自由运行
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

    # set_rate 在这个模型中不再有意义，因为时钟由数据驱动，而不是时间流逝驱动
    def set_rate(self, rate):
        pass


# --- 网络监控 (无变化) ---
class NetworkMonitor:
    def __init__(self): self.reset()
    def reset(self):
        self.lock = threading.Lock(); self.received_packets = 0; self.lost_packets = 0
        self.expected_seq = -1
    def record_packet(self, seq):
        with self.lock:
            if self.expected_seq == -1: self.expected_seq = seq
            if seq > self.expected_seq: self.lost_packets += seq - self.expected_seq
            self.expected_seq = seq + 1
            self.received_packets += 1
    def get_statistics(self):
        with self.lock:
            total = self.received_packets + self.lost_packets
            loss_rate = self.lost_packets / total if total > 0 else 0.0
            self.received_packets = 0; self.lost_packets = 0
            return {"loss_rate": loss_rate}

# --- 视频Jitter Buffer (无变化) ---
class VideoPacketJitterBuffer:
    def __init__(self, max_size=200):
        self.max_size = max_size
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.buffer = []  # 使用heapq作为优先队列
        self.expected_seq = -1

    def add_packet(self, packet, monitor):
        if len(packet) < 6: return  # 2字节序列号 + 4字节时间戳
        seq = int.from_bytes(packet[0:2], 'big')
        monitor.record_packet(seq)
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
                self.expected_seq = (self.expected_seq + 1) % (2**16)
                return packet
            elif seq < self.expected_seq: # 处理旧包或乱序
                heapq.heappop(self.buffer)
                return self.get_packet() # 递归获取下一个
            else: # 丢包发生
                seq, packet = heapq.heappop(self.buffer)
                self.expected_seq = (seq + 1) % (2**16)
                return packet

# --- 解码后帧的缓冲 (无变化) ---
class DecodedFrameBuffer:
    def __init__(self, buffer_size_ms=500):
        self.buffer_size_ms = buffer_size_ms
        self.reset()

    def reset(self):
        self.lock = threading.Lock()
        self.queue = deque()
        self.last_played_pts = -1

    def add_frame(self, frame): # frame是(pts, image_data)
        with self.lock:
            self.queue.append(frame)
            self.queue = deque(sorted(self.queue, key=lambda x: x[0]))

    def get_frame(self, target_pts_ms):
        with self.lock:
            if not self.queue or target_pts_ms == -1: return None
            best_frame = None
            for pts, img_data in self.queue:
                if pts <= target_pts_ms:
                    best_frame = (pts, img_data)
                else:
                    break
            if best_frame:
                self.last_played_pts = best_frame[0]
                while self.queue and self.queue[0][0] <= self.last_played_pts:
                    self.queue.popleft()
                return best_frame[1]
            return None

# --- 音频Jitter Buffer (无变化) ---
class AudioJitterBuffer:
    def __init__(self, max_size=200):
        self.max_size = max_size; self.silence = b'\x00' * (AUDIO_CHUNK * 2 * AUDIO_CHANNELS); self.reset()
    def reset(self): self.lock = threading.Lock(); self.buffer = []; self.expected_seq = -1
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
                heapq.heappop(self.buffer); self.expected_seq += 1
                return ts, payload
            elif seq < self.expected_seq:
                heapq.heappop(self.buffer); return self.get_chunk()
            else:
                self.expected_seq += 1; return None, self.silence
    def clear(self): self.reset()

# --- 线程函数 ---
def video_receiver_thread(sock, packet_buffer, monitor, running_flag):
    while running_flag.get('running'):
        try:
            packet_buffer.add_packet(sock.recv(65535), monitor)
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
            time.sleep(0.005)  # Jitter buffer在等待包
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
                # 打印解码错误有助于调试，但不要让它崩溃
                # print(f"[客户端-解码] 解码TS={ts}时发生错误: {e}")
                pass

        # 清理过时的重组缓冲区条目，防止内存泄漏
        # (如果一个分片帧的结尾丢失，它会永远留在缓冲区里)
        # 简单策略：只保留最新的几个时间戳
        if len(reassembly_buffer) > 10:
            oldest_ts = min(reassembly_buffer.keys())
            del reassembly_buffer[oldest_ts]


    print("[客户端-解码] 视频解码线程已停止。")


# --- [已修改] 音频播放线程 ---
def audio_player_thread(audio_buffer, clock, state_vars, running_flag):
    p = pyaudio.PyAudio(); stream = None
    try:
        stream = p.open(format=pyaudio.paInt16, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, output=True)
        while running_flag.get('running'):
            is_file_stream = state_vars.get('duration_sec', 0) > 0
            if is_file_stream and clock.is_paused:
                if stream.is_active(): stream.stop_stream()
                time.sleep(0.01); continue

            if not stream.is_active() and not clock.is_paused: stream.start_stream()

            pts_ms, chunk = audio_buffer.get_chunk()

            # 如果缓冲区为空，等待一下，避免CPU空转
            # 这也是处理网络卡顿和播放结束的关键
            if not chunk:
                time.sleep(0.01)
                continue

            # 只有音频线程可以启动和更新时钟
            if clock.get_time_ms() == -1 and pts_ms is not None:
                clock.start(pts_ms)

            # [关键修改] 用实际播放的音频块PTS来更新主时钟
            if pts_ms is not None:
                clock.update_time(pts_ms)

            # 播放音频
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
            stats = monitor.get_statistics(); stats['command'] = 'heartbeat'
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error: break


# --- QT6主窗口类 ---
class VideoStreamClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_connected = False; self.running_flag = {'running': False}; self.sockets = {}; self.threads = {}
        self.server_address = None; self.current_source = "无"; self.scale_mode = "fit"; self.last_frame = None
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

        self.init_ui()
        self.start_ui_updater()

    def init_ui(self):
        self.setWindowTitle("高级视频流客户端 (H.265版)")
        self.setGeometry(100, 100, 1000, 800)
        main_widget=QWidget(); self.setCentralWidget(main_widget); main_layout=QHBoxLayout(main_widget)
        left_panel=QWidget(); left_layout=QVBoxLayout(left_panel); conn_group=QWidget(); conn_layout=QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("服务器IP:")); self.ip_entry=QLineEdit("127.0.0.1"); conn_layout.addWidget(self.ip_entry)
        self.connect_btn=QPushButton("连接"); self.connect_btn.clicked.connect(self.toggle_connection)
        conn_layout.addWidget(self.connect_btn); left_layout.addWidget(conn_group)
        self.video_list=QListWidget(); left_layout.addWidget(QLabel("播放列表:")); left_layout.addWidget(self.video_list)
        self.play_btn=QPushButton("播放选中项"); self.play_btn.setEnabled(False); self.play_btn.clicked.connect(self.play_selected)
        left_layout.addWidget(self.play_btn); right_panel=QWidget(); right_layout=QVBoxLayout(right_panel)
        self.video_label=QLabel("请连接服务器并选择一个视频源"); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding); self.video_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        right_layout.addWidget(self.video_label, stretch=1); controls_widget=QWidget(); controls_layout=QVBoxLayout(controls_widget)
        self.progress_slider=QSlider(Qt.Horizontal); self.progress_slider.setEnabled(False); self.progress_slider.sliderMoved.connect(self.seek_slider_moved)
        self.progress_slider.sliderReleased.connect(self.seek_slider_released); controls_layout.addWidget(self.progress_slider)
        bottom_bar=QHBoxLayout(); self.play_pause_btn=QPushButton(); self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_pause_btn.setEnabled(False); self.play_pause_btn.clicked.connect(self.toggle_pause); bottom_bar.addWidget(self.play_pause_btn)
        self.time_label=QLabel("00:00 / 00:00"); bottom_bar.addWidget(self.time_label); bottom_bar.addStretch()
        bottom_bar.addWidget(QLabel("音量:")); self.volume_slider=QSlider(Qt.Horizontal); self.volume_slider.setRange(0,100); self.volume_slider.setValue(100)
        self.volume_slider.setMaximumWidth(150); self.volume_slider.valueChanged.connect(self.change_volume); bottom_bar.addWidget(self.volume_slider)
        bottom_bar.addWidget(QLabel("速率:")); self.speed_combo=QComboBox(); self.speed_combo.addItems(["1.0x"]); self.speed_combo.setCurrentIndex(0)
        self.speed_combo.setEnabled(False); bottom_bar.addWidget(self.speed_combo); controls_layout.addLayout(bottom_bar)
        right_layout.addWidget(controls_widget); main_layout.addWidget(left_panel, stretch=1); main_layout.addWidget(right_panel, stretch=3)
        self.status_bar=QStatusBar(); self.status_bar.showMessage("状态: 未连接"); self.setStatusBar(self.status_bar)
        self.video_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.video_label.customContextMenuRequested.connect(self.show_context_menu)
    def format_time(self, seconds): s=int(seconds); return f"{s // 60:02d}:{s % 60:02d}"

    def connect(self):
        server_ip = self.ip_entry.text()
        if not server_ip: self.status_bar.showMessage("错误: 请输入服务器IP地址"); return
        self.server_address = (server_ip, CONTROL_PORT); self.running_flag['running'] = True
        try:
            self.sockets = {'control': socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 'video': socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 'audio': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)}
            for s in ['video', 'audio']: self.sockets[s].settimeout(1.0)
            self.sockets['video'].bind(('', VIDEO_PORT)); self.sockets['audio'].bind(('', AUDIO_PORT))
            self.sockets['control'].settimeout(5); self.sockets['control'].sendto(json.dumps({"command": "get_list"}).encode(), self.server_address)
            data, _ = self.sockets['control'].recvfrom(2048); self.sockets['control'].settimeout(None)
            self.video_list.clear(); [self.video_list.addItem(item) for item in json.loads(data.decode())]
            self.start_threads(); self.is_connected = True; self.connect_btn.setText("断开"); self.play_btn.setEnabled(True)
            self.status_bar.showMessage("状态: 连接成功，请选择播放项")
        except Exception as e: self.status_bar.showMessage(f"连接失败: {str(e)}"); self.cleanup()

    def toggle_connection(self):
        if self.is_connected: self.disconnect()
        else: self.connect()

    def disconnect(self):
        if self.sockets.get('control') and self.server_address:
            try: self.sockets['control'].sendto(json.dumps({"command": "stop"}).encode(), self.server_address)
            except socket.error: pass
        self.cleanup()

    def start_ui_updater(self):
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui_tick)
        self.ui_timer.start(30) # UI更新频率无需太高

    # --- [已修改] UI更新逻辑 ---
    def update_ui_tick(self):
        now_ms = self.master_clock.get_time_ms()
        if self.player_status != self.PLAYER_STATE_PLAYING:
            return

        # 视频帧渲染
        if not self.master_clock.is_paused:
            if now_ms != -1:
                frame_data = self.decoded_frame_buffer.get_frame(now_ms)
                if frame_data is not None:
                    self.last_frame = frame_data
                    self.display_frame()

        # 进度条和时间标签更新
        is_file_stream = self.playback_state['duration_sec'] > 0
        if is_file_stream:
            # 如果已标记为播放结束，则不再更新
            if self.playback_state.get('playback_finished', False):
                return

            total_sec = self.playback_state['duration_sec']
            if now_ms != -1:
                current_sec = now_ms / 1000.0

                # [关键修改] 处理播放结束的情况
                if current_sec >= total_sec:
                    current_sec = total_sec
                    self.playback_state['playback_finished'] = True
                    self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                    print("[客户端] 视频播放结束。")

                if not self.progress_slider.isSliderDown():
                    if total_sec > 0:
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

    def cleanup(self):
        self.player_status = self.PLAYER_STATE_STOPPED
        if not self.running_flag.get('running'): return
        self.running_flag['running'] = False
        for thread in self.threads.values():
            if thread.is_alive(): thread.join(timeout=1.0)
        for sock in self.sockets.values(): sock.close()
        self.threads.clear(); self.sockets.clear(); self.is_connected = False
        self.current_source = "无"; self.last_frame = None
        self.connect_btn.setText("连接"); self.play_btn.setEnabled(False)
        self.status_bar.showMessage("状态: 未连接"); self.reset_playback_ui()

    def start_threads(self):
        thread_map = {
            'video_recv': (video_receiver_thread, (self.sockets['video'], self.video_packet_jitter_buffer, self.monitor, self.running_flag)),
            'video_decode': (video_decoder_thread, (self.video_packet_jitter_buffer, self.decoded_frame_buffer, self.master_clock, self.running_flag)),
            'audio_recv': (audio_receiver_thread, (self.sockets['audio'], self.audio_jitter_buffer, self.running_flag)),
            'audio_play': (audio_player_thread, (self.audio_jitter_buffer, self.master_clock, self.playback_state, self.running_flag)),
            'feedback': (feedback_sender_thread, (self.sockets['control'], self.server_address, self.monitor, self.running_flag))
        }
        for name, (target, args) in thread_map.items():
            self.threads[name] = threading.Thread(target=target, args=args, daemon=True)
            self.threads[name].start()

    def reset_playback_state(self):
        print("[客户端] 重置所有播放状态和缓冲区...")
        self.video_packet_jitter_buffer.reset()
        self.decoded_frame_buffer.reset()
        self.audio_jitter_buffer.clear() # audio buffer有自己的clear方法
        self.master_clock.reset()
        self.monitor.reset()
        self.last_frame = None
        self.playback_state['playback_finished'] = False # 重置播放结束标记

    def play_selected(self):
        selected_items=self.video_list.selectedItems();
        if not selected_items: self.status_bar.showMessage("提示: 请先选择一个视频"); return
        self.current_source = selected_items[0].text(); print(f"\n[客户端] 请求播放: {self.current_source}")
        self.reset_playback_state(); self.player_status=self.PLAYER_STATE_LOADING
        self.video_label.setText("加载中..."); self.status_bar.showMessage(f"状态: 正在请求播放 {self.current_source}...")
        threading.Thread(target=self.request_play_worker, daemon=True).start()

    def request_play_worker(self):
        try:
            self.sockets['control'].sendto(json.dumps({"command": "play", "source": self.current_source}).encode(), self.server_address)
            self.sockets['control'].settimeout(5.0); data, _ = self.sockets['control'].recvfrom(1024); self.sockets['control'].settimeout(None)
            response = json.loads(data.decode())
            if response.get("command") == "play_info": self.worker_signals.play_info_received.emit(response)
            else: self.worker_signals.play_failed.emit("错误: 服务器响应无效")
        except Exception as e: self.worker_signals.play_failed.emit(f"播放失败: {str(e)}")

    def on_play_info_received(self, response):
        self.player_status = self.PLAYER_STATE_PLAYING
        self.playback_state['duration_sec'] = response.get("duration", 0)
        self.playback_state['playback_finished'] = False # 确保新播放开始时标记为未结束
        self.setup_ui_for_playback(is_file_stream=(self.playback_state['duration_sec'] > 0))
        self.status_bar.showMessage(f"状态: 正在播放 {self.current_source}")

    def on_play_failed(self, error_message): self.player_status=self.PLAYER_STATE_STOPPED; self.status_bar.showMessage(error_message); self.reset_playback_ui()

    def toggle_pause(self):
        if self.playback_state.get('playback_finished', False): return # 播放结束后，暂停/播放按钮无效
        if self.master_clock.is_paused:
            self.master_clock.resume()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.master_clock.pause()
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def seek_slider_moved(self, value):
        if self.playback_state['duration_sec'] > 0:
            target_sec=self.playback_state['duration_sec']*(value/1000.0)
            self.time_label.setText(f"{self.format_time(target_sec)} / {self.format_time(self.playback_state['duration_sec'])}")

    def seek_slider_released(self):
        if self.playback_state['duration_sec'] == 0: return
        target_sec=self.playback_state['duration_sec']*(self.progress_slider.value()/1000.0)
        print(f"[客户端] 请求跳转到 {target_sec:.2f}s")
        self.reset_playback_state()
        try:
            self.sockets['control'].sendto(json.dumps({"command": "seek", "time": target_sec}).encode(), self.server_address)
            if self.master_clock.is_paused: self.toggle_pause()
        except Exception as e: self.status_bar.showMessage(f"跳转失败: {e}")

    def change_volume(self, value): self.playback_state['volume'] = value/100.0
    def change_speed(self, index): pass # set_rate in the new clock model is a no-op

    def reset_playback_ui(self):
        self.video_label.clear(); self.video_label.setText("请连接服务器并选择一个视频源"); self.time_label.setText("00:00 / 00:00")
        self.progress_slider.setValue(0); self.progress_slider.setEnabled(False); self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)); self.speed_combo.setEnabled(False)

    def setup_ui_for_playback(self, is_file_stream):
        self.progress_slider.setEnabled(is_file_stream); self.play_pause_btn.setEnabled(is_file_stream); self.speed_combo.setEnabled(False)
        if is_file_stream:
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)); self.progress_slider.setRange(0, 1000)
        else:
            self.time_label.setText("直播"); self.progress_slider.setValue(0)
            self.play_pause_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)); self.play_pause_btn.setEnabled(False)

    def show_context_menu(self, pos):
        menu=QMenu(self); actions={"自适应缩放 (Adapt)": "adapt", "按比例缩放 (Fit)": "fit", "原始大小 (Original)": "original"}
        for text, mode in actions.items():
            action=QAction(text, self, checkable=True, checked=(self.scale_mode == mode))
            action.triggered.connect(lambda checked, m=mode: self.set_scale_mode(m)); menu.addAction(action)
        menu.exec(self.video_label.mapToGlobal(pos))
    def set_scale_mode(self, mode): self.scale_mode=mode; self.display_frame()
    def closeEvent(self, event): self.disconnect(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamClient()
    window.show()
    sys.exit(app.exec())