import cv2
import socket
import time
import threading
import json
from shared_config import *

# --- 1. 自适应流控器 ---
class AdaptiveStreamController:
    """
    根据客户端反馈的网络状态，动态调整视频流参数。
    """
    def __init__(self):
        # 定义不同网络质量下的视频参数策略
        self.configs = {
            "good": {"resolution": (640, 480), "jpeg_quality": 85, "fec_enabled": False, "fps_limit": 30},
            "medium": {"resolution": (480, 360), "jpeg_quality": 60, "fec_enabled": True, "fps_limit": 20},
            "poor": {"resolution": (320, 240), "jpeg_quality": 40, "fec_enabled": True, "fps_limit": 15},
        }
        self.current_strategy_name = "good"  # 初始策略
        self.lock = threading.Lock()

    def get_current_strategy(self):
        """获取当前生效的视频参数策略"""
        with self.lock:
            return self.configs[self.current_strategy_name]

    def update_strategy(self, loss_rate):
        """
        根据丢包率更新策略。
        - 丢包率 < 3%: 网络良好
        - 3% <= 丢包率 < 10%: 网络中等
        - 丢包率 >= 10%: 网络差
        """
        new_strategy_name = "good"
        if loss_rate >= 0.1:
            new_strategy_name = "poor"
        elif loss_rate >= 0.03:
            new_strategy_name = "medium"

        with self.lock:
            if self.current_strategy_name != new_strategy_name:
                self.current_strategy_name = new_strategy_name
                print(f"[Controller] 丢包率: {loss_rate:.2%}, 切换策略至: {new_strategy_name.upper()}")

# --- 2. 控制信道处理 ---
def control_channel_handler(sock, controller, client_address_ref):
    """
    监听并处理来自客户端的控制信令。
    """
    print(f"[Control] 在端口 {CONTROL_PORT} 上监听客户端反馈...")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            # 当收到第一个包时，记录客户端地址，以便开始视频推流
            if client_address_ref['addr'] is None:
                print(f"[Control] 收到来自 {addr} 的连接请求，准备推流。")
                client_address_ref['addr'] = addr

            # 解析客户端反馈的JSON数据
            feedback = json.loads(data.decode())
            if 'loss_rate' in feedback:
                controller.update_strategy(feedback['loss_rate'])

        except (ConnectionResetError, socket.error) as e:
            print(f"[Control] 连接错误: {e}. 客户端可能已断开。")
            client_address_ref['addr'] = None # 重置客户端地址，停止推流
        except json.JSONDecodeError:
            print("[Control] 收到无法解析的控制信令。")
        except Exception as e:
            print(f"[Control] 处理控制信令时发生未知错误: {e}")

# --- 3. 前向纠错 (FEC) 生成 ---
def generate_fec_packets(data_packets: list) -> list:
    """
    为一组数据包生成FEC冗余包。
    使用简单的异或（XOR）方式生成。
    """
    fec_packets = []
    for i in range(0, len(data_packets), FEC_GROUP_SIZE):
        group = data_packets[i:i+FEC_GROUP_SIZE]
        if not group:
            continue

        # 确保组内所有包长度一致，便于异或操作
        max_len = max(len(p) for p in group)
        padded_group = [p.ljust(max_len, b'\0') for p in group]

        # 异或生成FEC数据
        fec_packet_data = bytearray(max_len)
        for p in padded_group:
            for j in range(max_len):
                fec_packet_data[j] ^= p[j]

        # 构建FEC包的头部
        # 包含：帧ID, FEC组内索引, is_fec=1等信息
        frame_id = int.from_bytes(data_packets[i][:4], 'big')
        fec_group_index = i // FEC_GROUP_SIZE
        header = frame_id.to_bytes(4, 'big') + \
                 fec_group_index.to_bytes(2, 'big') + \
                 (len(data_packets) // FEC_GROUP_SIZE).to_bytes(2, 'big') + \
                 (1).to_bytes(1, 'big') # 标记为FEC包

        fec_packets.append(header + fec_packet_data[9:]) # 替换头部
    return fec_packets

# --- 4. 主函数：视频捕获与推流 ---
def main():
    # 初始化UDP套接字
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock.bind((SERVER_HOST, CONTROL_PORT))

    # 初始化自适应控制器和客户端地址引用
    controller = AdaptiveStreamController()
    client_address_ref = {'addr': None} # 在线程间共享客户端地址

    # 启动控制信道监听线程
    control_thread = threading.Thread(
        target=control_channel_handler,
        args=(control_sock, controller, client_address_ref),
        daemon=True
    )
    control_thread.start()

    print(f"[Server] 服务器已启动。在 {SERVER_HOST}:{CONTROL_PORT} 等待客户端连接...")

    # 初始化视频捕获
    cap = cv2.VideoCapture(0) # 可替换为视频文件路径
    if not cap.isOpened():
        print("错误: 无法打开摄像头或视频文件。")
        return

    frame_id = 0
    while True:
        # 只有在确认客户端地址后才开始推流
        if not client_address_ref['addr']:
            time.sleep(0.5)
            continue

        client_addr = client_address_ref['addr']
        strategy = controller.get_current_strategy()
        start_time = time.time()

        # 读取和处理视频帧
        ret, frame = cap.read()
        if not ret:
            print("视频流结束。")
            break

        # 根据当前策略调整分辨率和JPEG质量
        frame = cv2.resize(frame, strategy["resolution"])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), strategy["jpeg_quality"]]
        _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
        frame_data = encoded_frame.tobytes()

        # 将帧数据分片打包
        packets = []
        total_packets = (len(frame_data) + PACKET_SIZE - 1) // PACKET_SIZE
        for i in range(total_packets):
            chunk = frame_data[i*PACKET_SIZE : (i+1)*PACKET_SIZE]
            # 构建数据包头部: 帧ID, 包索引, 总包数, is_fec=0
            header = frame_id.to_bytes(4, 'big') + \
                     i.to_bytes(2, 'big') + \
                     total_packets.to_bytes(2, 'big') + \
                     (0).to_bytes(1, 'big') # 标记为数据包
            packets.append(header + chunk)

        # 如果策略启用FEC，则生成并添加FEC包
        if strategy["fec_enabled"]:
            fec_packets = generate_fec_packets(packets)
            packets.extend(fec_packets)

        # 发送所有包
        for packet in packets:
            video_sock.sendto(packet, (client_addr[0], VIDEO_PORT))

        frame_id = (frame_id + 1) % (2**32 - 1) # 帧ID循环

        # 根据策略限制帧率
        elapsed = time.time() - start_time
        sleep_duration = max(0, (1.0 / strategy["fps_limit"]) - elapsed)
        time.sleep(sleep_duration)

    # 清理资源
    cap.release()
    video_sock.close()
    control_sock.close()
    print("[Server] 服务器已关闭。")

if __name__ == "__main__":
    main()
4