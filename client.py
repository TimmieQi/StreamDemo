import cv2
import socket
import numpy as np
import threading
import time
import json
from collections import defaultdict, deque

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

# --- 2. 帧重组与抖动缓冲 ---
class JitterBuffer:
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
        self.ready_queue = deque() # 存储已解码的完整帧 (frame_id, frame_data)
        self.lock = threading.Lock()
        self.buffer_delay = buffer_time_ms / 1000.0
        self.last_played_frame_id = -1

    def add_packet(self, packet: bytes, monitor: NetworkMonitor):
        """添加收到的UDP包到缓冲区"""
        if len(packet) < 9: return # 包头不完整

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

            # 决定是否播放下一帧
            # 简单的策略：如果队列中有帧，就播放最老的那一帧
            # 更优的策略可以考虑帧的时间戳和缓冲延迟
            frame_id, frame_data = self.ready_queue.popleft()
            self.last_played_frame_id = frame_id
            return frame_data

    def cleanup(self):
        """清理过时的包缓冲区"""
        with self.lock:
            cutoff_time = time.time() - 3 # 清理3秒前的旧缓冲区
            old_frame_ids = [fid for fid, buf in self.packet_buffers.items() if buf["timestamp"] < cutoff_time]
            for fid in old_frame_ids:
                print(f"[Cleanup] 清理过时的帧缓冲: {fid}")
                del self.packet_buffers[fid]

# --- 3. 视频接收与显示 ---
def video_receiver_thread(sock, jitter_buffer, monitor, running_flag):
    """接收视频数据包的线程"""
    while running_flag['running']:
        try:
            data, _ = sock.recvfrom(65535)
            jitter_buffer.add_packet(data, monitor)
        except socket.error:
            print("[Video] 套接字错误，接收线程退出。")
            break

def display_thread(jitter_buffer, running_flag):
    """显示视频帧的线程"""
    while running_flag['running']:
        frame_data = jitter_buffer.get_frame()
        if frame_data:
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow("Video Stream", frame)
            except Exception as e:
                print(f"[Display] 解码或显示帧时出错: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running_flag['running'] = False
            break
    cv2.destroyAllWindows()

# --- 4. 控制信令发送 ---
def feedback_sender_thread(sock, server_addr, monitor, running_flag):
    """定期向服务器发送网络状态反馈的线程"""
    while running_flag['running']:
        time.sleep(1) # 每秒发送一次反馈
        stats = monitor.get_statistics()
        try:
            sock.sendto(json.dumps(stats).encode(), server_addr)
        except socket.error as e:
            print(f"[Feedback] 发送反馈失败: {e}")

# --- 5. 主函数 ---
def main():
    # 提示用户输入服务器IP
    server_ip = input("请输入服务器的IP地址 (例如 192.168.1.100): ")
    server_address = (server_ip, CONTROL_PORT)

    # 初始化套接字
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock.bind(('', VIDEO_PORT)) # 绑定到所有接口的视频端口
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 初始化核心组件
    monitor = NetworkMonitor()
    jitter_buffer = JitterBuffer(buffer_time_ms=150) # 150ms的抖动缓冲
    running_flag = {'running': True}

    # 启动视频接收线程
    recv_thread = threading.Thread(target=video_receiver_thread, args=(video_sock, jitter_buffer, monitor, running_flag))
    recv_thread.daemon = True
    recv_thread.start()

    # 启动视频显示线程
    disp_thread = threading.Thread(target=display_thread, args=(jitter_buffer, running_flag))
    disp_thread.start()

    # 启动网络反馈发送线程
    feedback_thread = threading.Thread(target=feedback_sender_thread, args=(control_sock, server_address, monitor, running_flag))
    feedback_thread.daemon = True
    
    # 启动缓冲区清理线程
    cleanup_thread = threading.Thread(target=lambda: (
        time.sleep(5), jitter_buffer.cleanup()
    ), daemon=True)
    cleanup_thread.start()

    print("客户端已启动。正在连接到服务器...")
    # 发送一个初始包来“注册”到服务器
    control_sock.sendto(json.dumps({"status": "connect"}).encode(), server_address)
    feedback_thread.start() # 在发送第一个包后启动反馈

    print("连接成功。视频流应在几秒钟内开始。")
    print("在视频窗口按 'q' 键退出。")

    # 等待显示线程结束 (当用户按'q'时)
    disp_thread.join()
    
    print("正在关闭客户端...")
    running_flag['running'] = False
    video_sock.close()
    control_sock.close()
    # 等待其他线程优雅退出
    recv_thread.join(timeout=1)
    feedback_thread.join(timeout=1)

if __name__ == "__main__":
    main()
