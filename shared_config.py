# shared_config.py

# --- 网络配置 ---
# 服务器绑定IP地址。'0.0.0.0' 表示监听所有可用的网络接口，
# 这样局域网内的其他设备才能连接，实现远程测试。
# 客户端连接时需要使用服务器在局域网中的实际IP地址（例如 192.168.1.100）。
SERVER_HOST = '0.0.0.0'
VIDEO_PORT = 9999      # 视频数据传输端口
AUDIO_PORT = 9997      # 音频数据传输端口
CONTROL_PORT = 9998    # 控制信令传输端口 (例如: 网络状态反馈)

# [NEW] 自定义RTP头大小 (2B seq + 4B ts + 1B frag_info)
VIDEO_HEADER_SIZE = 7

# --- 视频流参数 ---
VIDEO_CODEC = 'hevc'   # 视频编码器: hevc (H.265) 或 h264

# --- 音频流参数 ---
# 减小CHUNK大小可以降低捕获延迟，但会增加计算量。
# 比如256 / 16000Hz = 16ms 的捕获延迟。
AUDIO_CHUNK = 256      # 每次读取的音频数据大小 (样本数)
AUDIO_FORMAT = 8        # paInt16, 16位音频
AUDIO_CHANNELS = 1      # 单声道
AUDIO_RATE = 16000      # 采样率 (Hz)