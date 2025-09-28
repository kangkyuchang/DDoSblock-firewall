import socket
import struct
import os
from scapy.all import IP, Packet
from cicflowmeter.flow import Flow
from cicflowmeter.features.context import PacketDirection
from config import SOCKET_PATH, IP_ADDRESS
from ai.ddos_detection_model import WINDOW_SIZE, predict

def set_direction(packet: Packet) -> PacketDirection:
    if packet["IP"].dst == IP_ADDRESS:
        return PacketDirection(1)
    return PacketDirection(2)

def recv_all(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("소켓 연결 손실")
        data += packet
    return data

if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)

print("연결 대기 중...")
conn, _ = server.accept()
print("연결")

received_flows : dict[str, Flow] = {}
features = ["init_fwd_win_byts", "fwd_pkt_len_mean", "pkt_len_mean", "ack_flag_cnt", "bwd_pkts_s",
            "totlen_fwd_pkts", "flow_iat_max", "fwd_pkt_len_max", "pkt_len_std", "pkt_len_max", 
            "flow_pkts_s", "pkt_size_avg", "flow_iat_mean", "pkt_len_var"]

try:
    while True:
        raw_len = recv_all(conn, 4)
        pkt_len = struct.unpack("!I", raw_len)[0]
        packet = IP(recv_all(conn, pkt_len))
        # print(f"Received packet of length: {pkt_len}")
        is_attack = 0
        packet_dst = packet["IP"].dst
        packet_src = packet["IP"].src
        # print(packet_src, packet_dst)
        if packet_dst == IP_ADDRESS:
            flow = received_flows.get(packet_src)
            if flow is not None:
                flow.add_packet(packet, PacketDirection(1))
            else:
                flow = Flow(packet, PacketDirection(1))
                received_flows[packet_src] = flow
                remove_count = len(received_flows) - WINDOW_SIZE
                while remove_count > 0:
                    remove_count -= 1
                    key = next(iter(received_flows))
                    del received_flows[key]
                is_attack = predict(received_flows)
            # print("예측: ", is_attack)
        else:
            flow = received_flows.get(packet_dst)
            if flow is not None:
                flow.add_packet(packet, PacketDirection(2))
        data = is_attack.to_bytes(4, "little")
        conn.sendall(data)
except Exception as e:
    print("Exception:", e)
finally:
    conn.close()
    server.close()
    os.remove(SOCKET_PATH)