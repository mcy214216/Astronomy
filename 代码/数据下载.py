import mmap
import os
from multiprocessing import Process

def child_process(shm):
    # 子进程写入数据
    shm.seek(0)
    shm.write(b"Hello, I am child process!")
    shm.close()

def main():
    # 创建匿名内存映射区域（兼容Windows的共享方式）
    shm = mmap.mmap(-1, 1024)  # 分配1024字节

    # 创建子进程
    p = Process(target=child_process, args=(shm,))
    p.start()
    p.join()  # 等待子进程结束

    # 父进程读取数据
    shm.seek(0)
    content = shm.read(shm.size()).split(b'\x00')[0]  # 去除空字节
    print("Parent received:", content.decode())

    # 关闭资源
    shm.close()

if __name__ == "__main__":
    main()