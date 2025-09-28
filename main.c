#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <netinet/if_ether.h>
#include <net/ethernet.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <pcap.h>
#include "rule.h"
#include <linux/netfilter.h>
#include <linux/netlink.h>
#include <linux/netfilter/nfnetlink.h>
#include <libnetfilter_queue/libnetfilter_queue.h>

#define SOCKET_PATH "./tmp/packet_socket"

Rule rules[MAX_RULES];
int rule_count = 0;

pcap_t *handle;
char errbuf[PCAP_ERRBUF_SIZE];
struct bpf_program fp;
char filter_exp[] = "ip";
bpf_u_int32 net;

int sockfd;

static int packet_callback(struct nfq_q_handle *qh, struct nfgenmsg *nfmsg, struct nfq_data *nfa, void *data) {
    struct nfqnl_msg_packet_hdr *ph = nfq_get_msg_packet_hdr(nfa);

    if(!ph) return nfq_set_verdict(qh, 0, NF_ACCEPT, 0, NULL);
    
    uint32_t id = ntohl(ph->packet_id);
    
    unsigned char *packet_data;
    int len = nfq_get_payload(nfa, &packet_data);
    if (len <= 0) {
        return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
    }

    uint32_t net_len = htonl(len);

    ssize_t ret;
    struct iphdr *ip_header = (struct iphdr *) packet_data;

    uint32_t packet_verdict;

    uint8_t protocol = ip_header->protocol;
    uint32_t receive;
    if (ip_header->version == 4 && protocol == IPPROTO_TCP || protocol == IPPROTO_UDP) {
        ret = write(sockfd, &net_len, sizeof(net_len));

        if (ret != sizeof(net_len)) {
            perror("write length failed");
            return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
        }

        ret = write(sockfd, packet_data, len);
        if (ret <= 0) {
            perror("write packet_data failed");
            return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
        }

        recv(sockfd, &receive, sizeof(receive), 0);
        packet_verdict = receive == 1 ? NF_DROP : NF_ACCEPT;
        // printf("%d\n", receive);
    }
    printf("%d\n", receive);
    return nfq_set_verdict(qh, id, packet_verdict, 0, NULL);
}

int add_rule(const char *src_ip, const char *dest_ip, int src_port, int dest_port, const char *protocol, const char *action) {
    if(rule_count >= MAX_RULES) {
        return -1;
    }

    Rule *rule = &rules[rule_count++];
    strncpy(rule->src_ip, src_ip, sizeof(rule->src_ip));
    strncpy(rule->dest_ip, dest_ip, sizeof(rule->dest_ip));
    rule->src_port = src_port;
    rule->dest_port = dest_port;
    strncpy(rule->protocol, protocol, sizeof(rule->protocol));
    strncpy(rule->action, action, sizeof(rule->action));

    printf("규칙 추가 성공");
    return 0;
}

void block_ip(const char *ip, int port, const char *protocol) {
    char buffer[100];
    sprintf(buffer, "iptables -I INPUT -s %s -p %s --sport %d -j DROP", ip, protocol, port);
    // system(buffer);
}

void unblock_ip(const char *ip, int port, const char *protocol) {
    char buffer[100];
    sprintf(buffer, "iptables -D INPUT -s %s -p %s --sport %d -j DROP", ip, protocol, port);
    // sytem(buffer);
}

void load_rule() {
    FILE *fp = fopen("./blocked_ip.txt", "r");
    if(fp != NULL) {
        char line[64];
        const char split[] = " \t\n";
        while(fgets(line, sizeof(line), fp)) {
            char *src_ip = strtok(line, split);
            int src_port = atoi(strtok(NULL, split));
            char *dest_ip = strtok(NULL, split);
            int dest_port = atoi(strtok(NULL, split));
            char *protocol = strtok(NULL, split);
            char *action = strtok(NULL, split);
            add_rule(src_ip, dest_ip, src_port, dest_port, protocol, action);
            if(strcmp(action, "BLOCK") == 0) {
                block_ip(src_ip, src_port, protocol);
            }
        }
    }
    fclose(fp);
}

void save_rule() {
    FILE *fp = fopen("./blocked_ip.txt", "r");
    char *src_ip, *dest_ip, *protocol, *action;
    int src_port, dest_port;
    for(int i = 0; i < rule_count; i++) {
        Rule *rule = &rules[i];
        src_ip = rule->src_ip;
        src_port = rule->src_port;
        dest_ip = rule->dest_ip;
        dest_port = rule->dest_port;
        protocol = rule->protocol;
        action = rule->action;
        fprintf(fp, "%s %d %s %d %s %s\n", src_ip, src_port, dest_ip, dest_port, protocol, action);
        if(strcmp(action, "BLOCK") == 0) {
            unblock_ip(src_ip, src_port, protocol);
        }
    }
    fclose(fp);
}

int main() {
    struct nfq_handle *h;
    struct nfq_q_handle *qh;
    struct sockaddr_un addr;
    
    system("iptables -A INPUT -i enp3s0 -j NFQUEUE --queue-num 1");
    system("iptables -A OUTPUT -o enp3s0 -j NFQUEUE --queue-num 1");

    h = nfq_open();

    qh = nfq_create_queue(h, 1, &packet_callback, NULL);

    nfq_set_mode(qh, NFQNL_COPY_PACKET, 0xffff);

    sockfd = socket(AF_UNIX, SOCK_STREAM, 0);

    if(sockfd < 0) {
        perror("socket 생성 실패");
        return 1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if(connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("연결 실패");
        close(sockfd);
        return 1;
    }

    load_rule();

    int fd = nfq_fd(h);
    char buf[4096];
    while(1) {
        int len = recv(fd, buf, sizeof(buf), 0);
        nfq_handle_packet(h, buf, len);
    }

    system("iptables -D INPUT -i enp3s0 -j NFQUEUE --queue-num 1");

    close(sockfd);

    save_rule();

    return 0;
}

