"""Build step 1: raw GET /v1/models round-trip to LM Studio via libc sockets."""

from net import tcp_connect, tcp_send, tcp_recv_all, tcp_close


def main() raises:
    var fd = tcp_connect(127, 0, 0, 1, 1234)
    print("connected, fd =", fd)

    var req = String(
        "GET /v1/models HTTP/1.1\r\n"
        "Host: localhost:1234\r\n"
        "Connection: close\r\n"
        "\r\n"
    )
    tcp_send(fd, req)

    var resp = tcp_recv_all(fd)
    tcp_close(fd)

    print("---- response (", resp.byte_length(), "bytes ) ----")
    print(resp)
