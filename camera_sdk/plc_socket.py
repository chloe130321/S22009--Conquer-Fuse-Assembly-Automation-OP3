from socket import *

class plc_socket():
    def __init__(self, host, port):
        self.HOST = host
        self.PORT = port
        self.conn = socket(AF_INET, SOCK_STREAM)
        

        try:
            self.conn.connect((self.HOST, self.PORT))
            print("Successfully connected to the server.")
            
        except ConnectionRefusedError:
            print("Connection refused. Unable to connect to the server.")
            
        except TimeoutError:
            print("Connection timeout. Unable to connect to the server.")
            
        except Exception as e:
            print("An error occurred while connecting:", str(e))


    def Send(self, register, value, bit=''):
        cmd = "WR " + str(register) + bit + " " + str(value) + "\x0D"
        self.conn.sendall(cmd.encode())
        return self.conn.recv(1024).decode()

    def Sends(self, register, num, datas, bit=''):
        self.__data = ""
        for x in datas:
            self.__data += " "+ str(x)
        cmd = "WRS " + str(register) + bit + " " + str(num) + self.__data + "\x0D"
        self.conn.sendall(cmd.encode())
        return self.conn.recv(1024).decode()

    def Get(self, register, bit=''):
        cmd = "RD " + register + str(bit) +"\x0D"

        self.conn.sendall(cmd.encode())
        return self.conn.recv(1024).decode().strip()


    def Gets(self, register, nums, bit=''):
        cmd = "RDS " + register + bit + ' ' + str(nums) +"\x0D"

        self.conn.sendall(cmd.encode())
        self.__list = []
        [self.__list.append(int(x)) for x in self.conn.recv(1024).decode().strip().split(" ")]
        return self.__list

    def Close(self):
        self.conn.close()





    # 資料格式如下:
    #  .U : 16位無符號十進位
    #  .S : 16位有符號十進位
    #  .D : 32位無符號十進位
    #  .L : 32位有符號十進位
    #  .H : 16位十六進位值數