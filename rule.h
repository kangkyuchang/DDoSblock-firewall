#define MAX_PATCKET_SIZE 65536
#define MAX_RULES 100

typedef struct {
    char src_ip[16];
    char dest_ip[16];
    int src_port;
    int dest_port;
    char protocol[8];
    char action[8];
} Rule;